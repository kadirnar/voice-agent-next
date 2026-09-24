# WebSocket transport (`van-ws/1`)

`voice_agent_next.transports.websocket` serves voice agents to browsers and backends over a
plain WebSocket. Each connection gets its own `AgentSession`. Audio travels as raw PCM16 in
binary frames. Control messages (handshake, barge-in, transcripts, state, metrics, playback
position, typed input) travel as JSON text frames. This follows the design recommended in
[research note 05](../research/05-latency-transports-production.md), §3.3 and §9.1.

Use it for server-to-server links, LAN and localhost apps, prototypes, and browsers on good
networks. WebSocket runs over TCP, so one lost packet stalls everything behind it
(head-of-line blocking). For browser and mobile clients on lossy networks, prefer WebRTC.

## Quick start

### Serve an agent: one session per connection

```python
import asyncio

from voice_agent_next import Agent, AgentSession
from voice_agent_next.transports.websocket import serve_websocket


async def main() -> None:
    server = await serve_websocket(
        lambda: AgentSession("openai/gpt-realtime"),  # or stt=/llm=/tts= for a cascade
        lambda: Agent("You are a helpful assistant.", greeting="Hi! How can I help?"),
        host="127.0.0.1",
        port=8765,
    )
    await server.serve_forever()  # Ctrl-C closes every session cleanly


asyncio.run(main())
```

### Try it offline in a browser

```bash
python examples/websocket_agent.py serve     # mock engine: no API keys, no downloads
# open http://127.0.0.1:8765/ (Chrome, Edge or Safari; headphones recommended)

python examples/websocket_agent.py client    # ...or talk to it from Python
python examples/websocket_agent.py serve --engine openai/gpt-realtime   # a real engine
```

The browser client is the single file [`examples/web/index.html`](../../examples/web/index.html).
The example server also serves it over plain HTTP on the WebSocket port. Browsers only allow
microphone access on `https://` pages and on `http://localhost` / `http://127.0.0.1`. To serve
the page from somewhere else, pass the agent address as `?ws=ws://host:8765/`.

### A single connection through `create_transport`

`create_transport({"type": "websocket", ...})` returns a standalone
`WebSocketServerTransport`. It listens on `host:port`, serves the **first** client that
completes the handshake, and ends when that client leaves. While that client is connected,
other clients are refused with close code 1013. To get transcripts, state and metrics on the
client, attach a `SessionBridge`:

```python
from voice_agent_next.transports import create_transport
from voice_agent_next.transports.websocket import SessionBridge

transport = create_transport({"type": "websocket", "host": "127.0.0.1", "port": 8765})
bridge = SessionBridge(session, transport)
try:
    await session.run(agent, transport)  # waits for a client; returns when it leaves
finally:
    await bridge.aclose()
```

## Protocol `van-ws/1`

One WebSocket connection carries one agent session. The URL path is not part of the protocol,
so servers may use it for routing or authentication.

* **Binary frames** carry audio: raw PCM, signed 16-bit little-endian, with no header.
* **Text frames** carry control messages. Each one is a JSON object with a string `type`.

```
client                                            server
  | --- hello {sample_rate: 16000, ...} ----------> |
  | <-------------- ready {output_sample_rate} ---- |  the session starts (greeting...)
  | === PCM16 @16 kHz, 20 ms binary frames =======> |
  | <======= PCM16 @24 kHz, <= 20 ms binary frames = |
  | <---------------- state / transcript / metrics  |
  | --- playback {position_ms} -------------------> |  ~every 100 ms while playing
  | <----------------------------------- clear ---- |  the user barged in
  | --- text {text} ------------------------------> |  typed input
  | --- (close) ----------------------------------> |  the session ends
```

### Handshake

The client's first message must be `hello`, and it must arrive within `hello_timeout` (10 s by
default):

```json
{"type": "hello", "protocol": "van-ws/1", "codec": "pcm_s16le", "sample_rate": 16000,
 "channels": 1, "output_sample_rate": 24000, "framing": "binary",
 "metadata": {"user": "ada"}}
```

| Field | Type | Default | Meaning |
|---|---|---|---|
| `type` | `"hello"` | required | |
| `protocol` | string | `"van-ws/1"` | Must be `"van-ws/1"` if present |
| `codec` | string | `"pcm_s16le"` | Only 16-bit little-endian PCM (`"pcm16"` and `"s16le"` are accepted aliases) |
| `sample_rate` | int, 8000–48000 | server's `input_sample_rate` (16000) | Sample rate of the client's audio |
| `channels` | 1 or 2 | 1 | Channels of the client's audio, interleaved (down-mixed by the server) |
| `output_sample_rate` | int, 8000–48000 | server's `output_sample_rate` (24000) | Preferred sample rate of the agent's audio |
| `framing` | `"binary"` or `"base64"` | `"binary"` | How the server sends agent audio (see [base64 mode](#base64-compatibility-mode)) |
| `metadata` | object | none | Free-form application data, available to the server as `transport.hello["metadata"]` |

The server answers `ready`. It is always the first message the server sends:

```json
{"type": "ready", "protocol": "van-ws/1", "session_id": "ws_3f9a1c0b7d2e", "codec": "pcm_s16le",
 "sample_rate": 16000, "channels": 1, "output_sample_rate": 24000, "output_channels": 1,
 "framing": "binary", "frame_ms": 20}
```

`sample_rate` and `channels` echo the accepted input format. `output_sample_rate` and
`output_channels` give the format of the agent's audio, which is always mono. Clients must use
these values, because they can differ from what was requested. `frame_ms` is the longest agent
audio message the server sends.

If the `hello` is missing, malformed or rejected, the server sends an `error` and closes the
connection with code 1002. A client may start streaming audio right after `hello`. The server
buffers it until the session runs.

### Audio

**Client to server.** Send PCM in the format announced in `hello`. Frames of 20 ms are
recommended: 640 bytes at 16 kHz mono. Any size works up to the message limit (1 MiB by
default), and a frame may even split a sample: an odd trailing byte is carried over to the next
frame. Keep streaming while the user is silent, and send zeros when muted. Engines detect the
end of the user's turn from that silence.

**Server to client.** The agent's speech arrives as mono PCM at `output_sample_rate`, in
messages of at most `frame_ms`. It is paced in real time, with about 150 ms of look-ahead.
Play it as it arrives (a small jitter buffer is fine), and do not assume a fixed message size.

#### base64 compatibility mode

With `"framing": "base64"` the server sends agent audio as JSON instead of binary frames:
`{"type": "audio", "data": "<base64 PCM>"}`. The server always accepts client audio in either
form, whatever the framing. Base64 adds 33% overhead, so use it only for clients that cannot
handle binary frames.

### Server-to-client messages

| `type` | Fields | When |
|---|---|---|
| `ready` | see [Handshake](#handshake) | Once, first |
| `clear` | none | The user interrupted the agent. **Drop all agent audio not played yet, immediately.** Audio received after `clear` belongs to the next response. |
| `transcript` | `role`, `item_id`, `text`, `final`, plus the role-specific fields below | User and agent speech as text |
| `state` | `agent`, `user` | Every change of either state |
| `metrics` | `kind`, `data` | Per-turn latency and component metrics |
| `error` | `code`, `message`, `fatal` (optional) | See [Errors](#errors-and-close-codes) |
| `audio` | `data` (base64) | Agent audio in base64 mode only |

**`transcript`**

* `role: "user"`: `text` is the current hypothesis. Interim transcripts (`final: false`)
  replace each other, and `final: true` ends the item. `language` is included when known.
* `role: "assistant"`: streamed as the agent speaks. Each message has `delta` (the new text),
  `text` (the full text so far) and `response_id`. A last message with `final: true` carries
  the complete text. After a barge-in, that message instead carries `interrupted: true`,
  `played_ms`, and only the text the user actually heard.
* Text the user types (see `text` below) is not echoed back.

```json
{"type": "transcript", "role": "user", "item_id": "item_1", "text": "what's the weather", "final": true}
{"type": "transcript", "role": "assistant", "item_id": "item_2", "response_id": "resp_7", "delta": "It is sunny.", "text": "It is sunny.", "final": false}
{"type": "transcript", "role": "assistant", "item_id": "item_2", "response_id": "resp_7", "text": "It is", "final": true, "interrupted": true, "played_ms": 640}
```

**`state`** sends both states on every change. `agent` is one of `initializing`, `listening`,
`thinking`, `speaking` or `closed`. `user` is `listening` or `speaking`.

**`metrics`**: `kind` is one of `turn`, `engine`, `stt`, `llm`, `tts`, `vad` or `eot`. `data`
holds the fields of the matching dataclass in `voice_agent_next.metrics`, with durations in
seconds. `kind: "turn"` carries `voice_to_voice`, `end_of_turn_delay`, `response_ttfb`,
`agent_speech_duration`, `interrupted` and `tool_calls`.

On the server, applications can send their own messages with `transport.send_message({...})`.

### Client-to-server messages

| `type` | Fields | Meaning |
|---|---|---|
| `hello` | see [Handshake](#handshake) | First message, once |
| `playback` | `position_ms` | Playback cursor (below). `mark` with `played_ms` is accepted as an alias. |
| `text` | `text` | A typed user message. The agent answers it. |
| `audio` | `data` (base64) | Client audio as JSON, as an alternative to binary frames |
| *anything else* | any | Passed to the application as a transport `"message"` event |

### Playback position and barge-in

The agent must know what the listener actually heard, both to cut its reply at the right word
and to know when it has finished speaking. The client reports its **playback cursor**:

```
position_ms = (agent audio received so far) - (agent audio still queued for playback)
```

Both terms are milliseconds of agent audio counted over the whole connection. Audio dropped
because of `clear` counts as passed, so after a `clear` the cursor jumps to the end of
everything received. Send a report:

* every 100–250 ms while audio is playing;
* when the playback queue drains;
* right after handling `clear`.

```
server sent 2000 ms           client: received 2000, queued 600   -> position_ms 1400
server sends clear            client drops its 600 queued ms      -> position_ms 2000
server sent 500 ms more       client: received 2500, queued 500   -> position_ms 2000
```

The server uses these reports for `buffered_duration()` and `wait_for_playout()`. Each report
re-anchors the estimate, which is then extrapolated in real time until the next one. Reports
that predate the last `clear` are ignored. A client that never reports is modelled as a
speaker that starts playing each chunk the moment it is sent. The session truncates an
interrupted reply using its own real-time playout clock, and the reports make the transport's
view match the real device.

On barge-in, the server sends `clear`, then a final `transcript` for the interrupted item with
`interrupted: true`, and then `state` changes.

### Errors and close codes

| `code` | Fatal | Close code | Meaning |
|---|---|---|---|
| `bad_hello` | yes | 1002 | Missing or malformed `hello`, an invalid field, or no `hello` in time |
| `unsupported_protocol` | yes | 1002 | `protocol` is not `van-ws/1` |
| `unsupported_codec` | yes | 1002 | `codec` is not `pcm_s16le` |
| `server_busy` | yes | 1013 | Another client holds a standalone transport, or `max_sessions` is reached. Try again later. |
| `internal_error` | yes | 1011 | The session could not be created or started (e.g. a missing API key) |
| `session_error` | per `fatal` | 1000 when the session then ends | The session reported an error (engine, tool...) |
| `invalid_message` | no | none | A malformed message after the handshake. The connection stays open. |
| `text_failed` | no | none | Typed input could not be processed |

Either side may close at any time. When the client closes, the session ends with reason
`user_disconnected`. When the session ends (the agent hangs up, or a fatal engine error), the
server closes with 1000. Shutting the server down closes every connection with 1001.
`message` fields contain exception text, which helps during development.

### Versioning

`van-ws/1` stays compatible for its whole lifetime. New optional fields and new message types
may be added, and **both sides must ignore unknown fields and message types**. A breaking
change will be published as `van-ws/2`. A server rejects protocols it does not speak with
`unsupported_protocol`.

## Python API

### `WebSocketServerTransport`

A `Transport` for one connection, with capabilities `messages` and `playback_position`. It has
two modes:

* **Per connection:** `WebSocketServerTransport(websocket)` wraps a connection accepted by a
  `websockets` server. This is what `serve_websocket` does.
* **Standalone:** `WebSocketServerTransport(host=..., port=...)`, or `create_transport`.
  `start()` listens and waits for the first client. `listen()` starts listening without
  waiting, so `port=0` resolves to a real port.

| Option | Default | Meaning |
|---|---|---|
| `host`, `port` | `"127.0.0.1"`, `8765` | Listening address in standalone mode (`port=0` picks a free port) |
| `input_sample_rate` | 16000 | Input rate assumed when `hello` omits `sample_rate` |
| `output_sample_rate` | 24000 | Agent audio rate unless `hello` asks for another one |
| `frame_duration` | 0.02 | Longest agent audio message, in seconds |
| `hello_timeout` | 10.0 | Seconds to wait for `hello` |
| `flush_timeout` | 1.0 | On close, seconds to wait for queued control messages to be sent |
| `serve_options` | `{}` | Extra `websockets.asyncio.server.serve` arguments in standalone mode |

Attributes: `hello` (the client's hello), `session_id`, `path` (request path with query string),
`url`, `framing` and `connected`. Events: `"connected"`, `"disconnected"` and `"message"` (a
client message of unknown type). `write_audio()` never blocks. A writer task sends queued
messages in order, so `clear_audio()` also drops audio that has not reached the socket.
`send_message_nowait()` is the synchronous form of `send_message()`.

### `serve_websocket()` / `WebSocketAgentServer`

```python
server = await serve_websocket(
    session_factory,
    agent_factory,
    host="127.0.0.1",
    port=8765,
    max_sessions=20,
    origins=["https://app.example.com"],
)
print(server.url, server.port, server.sessions)
await server.serve_forever()  # or: async with WebSocketAgentServer(...) as server: ...
```

For every connection, the server:

1. performs the handshake;
2. calls both factories;
3. attaches a `SessionBridge`, unless `forward_events=False`;
4. runs the session until either side hangs up.

Each factory may be a plain function or a coroutine function. It either takes no argument or
takes the connection's `WebSocketServerTransport`, which lets it read `transport.hello` or
`transport.path`:

```python
def agent_factory(transport: WebSocketServerTransport) -> Agent:
    meta = transport.hello.get("metadata") or {}
    return Agent("You are a concierge.", language=meta.get("language"))
```

Options: `max_sessions` (refuse extra clients with 1013) and `forward_events`, plus the
per-connection transport options (`input_sample_rate`, `output_sample_rate`, `frame_duration`,
`hello_timeout`). Any other keyword goes to `websockets.asyncio.server.serve`, for example
`ssl`, `origins`, `process_request`, `ping_interval` or `max_size`. Compression is off by
default, because PCM barely compresses and deflate adds latency.

### `SessionBridge`

`SessionBridge(session, transport, metrics=True, errors=True, text_input=True)` maps session
events to the messages above. It turns `user_transcript`, `agent_transcript` and `interrupted`
into `transcript`; state changes into `state`; `metrics` into `metrics`; and `error` into
`error`. It answers `text` messages with `session.generate_reply()`. It works with any
transport that has a message channel. Call `aclose()` to detach it.

## Browser client notes

[`examples/web/index.html`](../../examples/web/index.html) is a dependency-free reference
client:

* It calls `getUserMedia` with `echoCancellation`, `noiseSuppression` and `autoGainControl`.
* An AudioWorklet resamples the microphone from the device rate to 16 kHz (windowed-sinc
  low-pass plus interpolation) and posts 20 ms PCM16 frames.
* A second AudioWorklet queues agent audio, resamples it to the device rate, drops its queue on
  `clear`, and reports its cursor as `playback` every 100 ms while playing.
* The worklet code is loaded from a Blob URL, so the page stays one file.

Echo cancellation matters: without it, the agent hears itself and interrupts itself. Chrome,
Edge and Safari cancel WebAudio output well. Firefox's echo canceller is weaker, so use
headphones there.

## Security and deployment

* The default `host` is `127.0.0.1`. To accept remote clients, bind `0.0.0.0` behind TLS:
  pass `ssl=` or terminate TLS at a reverse proxy. Pages served over `https://` can only
  connect to `wss://` URLs.
* Browsers send an `Origin` header. Pass `origins=[...]` so that other websites cannot open
  sessions from your users' browsers (cross-site WebSocket hijacking). Each session may hold a
  paid engine connection.
* Authenticate before the upgrade in `process_request`, using a token in the query string or a
  cookie. Alternatively, authenticate in a factory with `transport.hello["metadata"]`: raising
  there refuses the client with `internal_error`.
* Use `max_sessions` to bound concurrent sessions.
* Proxies must pass WebSocket upgrades through without buffering. The server pings every 20 s,
  so idle timeouts above that keep connections alive.

## Limitations

* TCP has no loss concealment. A lost packet delays everything behind it, so use WebRTC on
  lossy networks.
* Audio is PCM only, with no Opus. That is about 256 kb/s upstream at 16 kHz and 384 kb/s
  downstream at 24 kHz.
* There is no reconnection or session resumption: a new connection starts a new session.
* There are no `pause`/`resume` messages (`capabilities.pause` is false) and no DTMF.
