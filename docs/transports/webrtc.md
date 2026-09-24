# WebRTC transport (`van-webrtc/1`)

`voice_agent_next.transports.webrtc` serves voice agents to browsers and mobile apps over
WebRTC, peer to peer, using [aiortc](https://github.com/aiortc/aiortc). Each peer gets its own
`AgentSession`. Audio is Opus over UDP (SRTP). Events (transcripts, state, metrics, typed
input) travel on a WebRTC data channel as the same JSON messages the
[WebSocket transport](websocket.md) uses. Signalling is one HTTP request: the client POSTs its
SDP offer and gets the answer back.

Use it for clients that cross the internet, as recommended in
[research note 05](../research/05-latency-transports-production.md), §3.1–3.2:

* **Loss.** A lost 20 ms packet is concealed. It does not stall the stream, unlike TCP.
* **Jitter.** The browser runs an adaptive jitter buffer.
* **Echo.** The browser's echo cancellation, noise suppression and gain control apply to the
  microphone before it is encoded.
* **Bandwidth.** Opus needs about 32 kb/s, where PCM over WebSocket needs about 256 kb/s.

For server-to-server links and telephony, use the WebSocket or telephony transports.

```bash
pip install 'voice-agent-next[webrtc]'   # aiortc (pure-Python wheel) + PyAV (bundles libopus)
```

## Quick start

### Serve an agent: one session per peer

```python
import asyncio

from voice_agent_next import Agent, AgentSession
from voice_agent_next.transports.webrtc import serve_webrtc


async def main() -> None:
    server = await serve_webrtc(
        lambda: AgentSession("openai/gpt-realtime"),  # or stt=/llm=/tts= for a cascade
        lambda: Agent("You are a helpful assistant.", greeting="Hi! How can I help?"),
        host="127.0.0.1",
        port=8080,
        ice_servers=["stun:stun.l.google.com:19302"],  # behind NAT; omit on a LAN
    )
    await server.serve_forever()  # Ctrl-C hangs up every peer cleanly


asyncio.run(main())
```

### Try it offline in a browser

```bash
python examples/webrtc/webrtc_agent.py serve     # mock engine: no API keys, no downloads
# open http://127.0.0.1:8080/ (Chrome, Edge or Safari; headphones recommended)

python examples/webrtc/webrtc_agent.py client --output reply.wav   # ...or a Python peer
python examples/webrtc/webrtc_agent.py serve --engine openai/gpt-realtime   # a real engine
```

The browser client is the single file
[`examples/webrtc/index.html`](../../examples/webrtc/index.html), which the example server
serves at `/`. Browsers only allow microphone access on `https://` pages and on
`http://localhost` / `http://127.0.0.1`. For remote browsers, pass `--certfile/--keyfile`
or put the server behind a TLS proxy. To serve the page from another origin, pass
`?server=https://agent.example.com/` and allow that origin with `--cors`.

### A single peer through `create_transport`

`create_transport({"type": "webrtc", ...})` returns a standalone `WebRTCTransport`. Its
`start()` serves the signalling endpoint on `host:port`, answers the **first** offer, and waits
until that peer is connected. The transport, and so the session, ends when that peer leaves.
Later offers get `503` while it is connected. To get transcripts, state and metrics on the
data channel, attach a `SessionBridge`:

```python
from voice_agent_next.transports import create_transport
from voice_agent_next.transports.websocket import SessionBridge

transport = create_transport(
    {"type": "webrtc", "port": 8080, "signaling_options": {"index_html": page}}
)
bridge = SessionBridge(session, transport)
try:
    await session.run(agent, transport)
finally:
    await bridge.aclose()
```

### Mount signalling in your own web app

The built-in signalling server is a small HTTP/1.1 server on `asyncio` streams. To serve
the offer from FastAPI, aiohttp or Starlette instead, alongside your authentication, pass
`serve_http=False` and call `handle_offer()` from your route:

```python
server = WebRTCAgentServer(
    make_session, make_agent, serve_http=False, ice_servers=["stun:stun.l.google.com:19302"]
)


@app.post("/offer")
async def offer(request: dict, user=Depends(current_user)) -> dict:
    try:
        return await server.handle_offer({**request, "metadata": {"user": user.id}})
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except TransportError as exc:  # full or shutting down
        raise HTTPException(503, str(exc))


@app.get("/config")
async def config() -> dict:
    return server.ice_config
```

Remember to `await server.aclose()` on shutdown.

## Protocol `van-webrtc/1`

### Signalling

| Request | Response |
|---|---|
| `GET /config` | `200 {"iceServers": [...], "iceTransportPolicy": "all" \| "relay"}`: pass it to `new RTCPeerConnection(...)` |
| `POST /offer` with `{"type": "offer", "sdp": "...", "metadata": {...}}` | `200 {"type": "answer", "sdp": "...", "session_id": "rtc_..."}` |
| `GET /` | `index_html`, if configured |
| `OPTIONS *` | `204`, the CORS preflight |

Errors are JSON `{"error": "..."}`:

* `400`: malformed JSON, not an offer, or no `m=audio` section.
* `404` / `405`: unknown path or method.
* `413`: body over 256 KiB.
* `503`: `max_sessions` reached or the server is shutting down.

Each request is served on its own connection (`Connection: close`). CORS headers are sent only
for origins listed in `cors_origins` (`"*"` allows any).

**The client must finish ICE gathering before it POSTs the offer.** aiortc does not support
trickle ICE, so the offer has to carry the client's candidates. Wait for
`iceGatheringState === "complete"`; the demo gives up after 2 s and sends what it has. The
answer carries the server's candidates. `metadata` is free-form: it is available as
`transport.offer["metadata"]` in the session and agent factories.

**Client offer requirements:**

* one audio m-section, sendrecv so the agent can answer on it (`pc.addTrack(micTrack)`);
* optionally, a data channel, which is **pre-negotiated** on both sides:
  `pc.createDataChannel("van", {negotiated: true, id: 0})`. Without it, the audio works but
  no events are exchanged.

### Audio

* **Codec.** Opus at 48 kHz. The browser negotiates it by default; aiortc also offers
  PCMU/PCMA.
* **User audio.** Decoded to 48 kHz, downmixed to mono and resampled to
  `input_sample_rate` (default 16 kHz) with `StreamResampler`. Each frame is timestamped when
  it arrives.
* **Agent audio.** `write_audio()` takes `output_sample_rate` audio (default 48 kHz, so
  nothing is resampled) and queues it. The outbound track pulls 20 ms frames in real time.
  When the queue is empty or paused it sends silence, so the RTP clock never stalls. If the
  event loop stalls for more than 200 ms, the track's clock restarts instead of bursting.
* **Echo cancellation.** The browser cancels echo; nothing is needed on the server. Chrome,
  Edge and Safari do it well. Firefox's echo canceller is weaker, so use headphones there.

### Data channel messages

Server to client (one JSON object per message):

| `type` | Fields | When |
|---|---|---|
| `ready` | `protocol` (`"van-webrtc/1"`), `session_id`, `codec` (`"opus"`), `sample_rate`, `output_sample_rate` | Once, when the channel opens |
| `clear` | none | The user interrupted the agent. The server has already stopped sending, so it is informational. |
| `transcript`, `state`, `metrics`, `error` | as in [`van-ws/1`](websocket.md#server-to-client-messages) | |

Client to server:

| `type` | Fields | Meaning |
|---|---|---|
| `playout` | `delay_ms` | The client's playout delay (below) |
| `text` | `text` | A typed user message. The agent answers it. |
| `bye` | none | Hang up now, before the DTLS close arrives |
| *anything else* | any | Passed to the application as a transport `"message"` event |

Messages sent before the channel opens are queued, up to 256, and flushed on open. Invalid
messages get an `error` with `code: "invalid_message"`.

### Playback position and barge-in

After a barge-in, the session needs to know how much of the agent's answer the user actually
heard. It uses that to truncate the transcript and the engine's context. The transport reports
this through `buffered_duration()` (`capabilities.playback_position` is true):

```text
buffered_duration = audio still queued for the encoder + playout_delay, while it is draining
```

* **Queued audio.** The part the server has not sent yet is known exactly. `clear_audio()`
  drops it at once, so the peer receives no more of the interrupted answer.
* **In-flight audio.** Audio that was sent but not yet heard is on the network, in the
  client's jitter buffer or in its output device. That is `playout_delay`: 60 ms until the
  client reports otherwise. The demo page reports it every second as
  `{"type": "playout", "delay_ms": ...}`, computed from `getStats()`: the average
  jitter-buffer delay over the last interval, plus half the round-trip time, plus about 20 ms
  for the output device. A browser cannot flush its jitter buffer, so up to `playout_delay` of
  audio already sent is still heard after a `clear`. The truncation accounts for it.
* **`pause_audio()` / `resume_audio()`.** Pausing sends silence and keeps the queue, which the
  session uses for false-interruption recovery (`capabilities.pause` is true).

## ICE, STUN and TURN

* **`ice_servers`** configures the server's peer connection. It accepts URL strings or
  `{"urls", "username", "credential"}` mappings. The default is none: host candidates only,
  which is enough on a LAN or when the server has a public IP address. Behind NAT, add a STUN
  server. aiortc uses at most **one STUN and one TURN** server (`turn:` over UDP or TCP,
  `turns:` over TCP).
* **`client_ice_servers`** is what `GET /config` gives clients. It defaults to `ice_servers`.
  Use it to hand out short-lived TURN credentials, which you can also build per request if you
  mount `handle_offer` yourself.
* **`ice_transport_policy="relay"`** uses only TURN relay candidates on both peers: clients
  receive `iceTransportPolicy: "relay"`, and the server gathers only relay candidates. This is
  the fix for P2P from cloud VPCs, where unreachable private candidates add seconds to
  connection setup (research note §3.2). It needs a TURN server. aiortc has no public API for
  this, so the server sets aioice's transport policy on its ICE gatherers; if a future aiortc
  release changes those internals, the offer fails loudly rather than silently gathering
  everything.
* **mDNS candidates.** Browsers hide their host addresses behind `<uuid>.local` names, which
  servers in containers usually cannot resolve. The transport removes `a=end-of-candidates`
  from offers. Otherwise aioice would give up on a component with no resolvable candidate, and
  the answer would carry no candidates at all. Instead, the browser's connectivity checks reach
  the server's candidates, and ICE learns the browser's address from them (peer-reflexive
  candidates). A peer that never connects is dropped after `connect_timeout` (default 20 s).
* **Ports.** aiortc binds ephemeral UDP ports for each peer. A firewall must allow them, or
  use TURN.

## Python API

### `WebRTCTransport`

```python
WebRTCTransport(
    *, host="127.0.0.1", port=8080,          # standalone signalling address (0 = any free port)
    ice_servers=None, ice_transport_policy="all",
    input_sample_rate=16_000, output_sample_rate=48_000,
    playout_delay=0.06,                      # initial client playout delay estimate, seconds
    connect_timeout=20.0, flush_timeout=0.5,
    signaling_options=None,                  # standalone: index_html, ssl, cors_origins, client_ice_servers
)
```

* `accept_offer(sdp)` → `{"type": "answer", "sdp", "session_id"}`. It is called once per
  transport: renegotiation is not supported.
* `start()` waits for ICE and DTLS to connect. In standalone mode it first serves signalling
  and waits for the first offer.
* Other members: `connected`, `session_id`, `offer`, `pc` (the `RTCPeerConnection`),
  `playout_delay`, `sent_duration`, `wait_disconnected()`.
* It emits `"connected"`, `"disconnected"` and `"message"`.
* It disconnects when the peer connection fails or closes, when the data channel or the
  inbound track ends, or on `bye`. `aclose()` flushes queued data-channel messages (up to
  `flush_timeout`), stops the track and closes the peer connection.

### `serve_webrtc()` / `WebRTCAgentServer`

```python
WebRTCAgentServer(
    session_factory, agent_factory, *, host="127.0.0.1", port=8080,
    ice_servers=None, client_ice_servers=None, ice_transport_policy="all",
    max_sessions=None, forward_events=True,
    index_html=None, cors_origins=(), ssl=None, serve_http=True,
    **transport_options,                     # WebRTCTransport options
)
```

* Factories are called once per peer, either with no arguments or with the peer's
  `WebRTCTransport`. They may be coroutine functions.
* `handle_offer(request)` answers one offer and starts its session. This is the hook for your
  own web framework.
* `sessions` and `transports` list the live peers. `aclose()` hangs up every peer and waits
  for their sessions.

### Helpers

* `paced_audio_track(AudioPlayout())` is an aiortc track that sends queued audio in real
  time. Python clients use it as a microphone; see the example's `client` command and the
  tests.
* `av_frame_to_audio()` converts a decoded PyAV frame to a mono `AudioFrame`.
* `normalize_ice_servers()` converts the ICE server formats above to `RTCIceServer` dicts.

## Measured (localhost, two aiortc peers)

* **Connection setup.** About 30 ms from applying the answer to "connected".
* **One-way audio latency**, from `write_audio` / microphone push to the first decoded frame
  on the other side: about 90 ms server → peer and about 115 ms peer → server, median of 10
  runs. Most of it is aiortc's receive-side jitter buffer and 20 ms packetization. The
  peer → server direction also includes resampling to 16 kHz.
* **Browsers.** They add their own adaptive jitter buffer, typically 20–80 ms on good
  networks, and report it through `playout`.

With the mock engine, the example measured 711 ms voice-to-voice, including the mock's 300 ms
response delay and its VAD end-of-speech silence.

## Security and deployment

* Browsers need HTTPS for the microphone, except on localhost. Pass `ssl=` or terminate TLS at
  a proxy. Media is always encrypted (DTLS-SRTP).
* Each session may hold a paid engine connection. Authenticate offers before answering them:
  mount `handle_offer` behind your auth, or check `transport.offer["metadata"]` in a factory,
  since raising there ends that peer's session. Bound concurrency with `max_sessions`.
* Only list trusted origins in `cors_origins`.
* The signalling server is intentionally minimal. It reads one request per connection with a
  10 s timeout, 16 KiB of headers, a 256 KiB body, and no chunked bodies. Put a real proxy in
  front of it for internet-facing deployments, or mount `handle_offer` in your framework.
* In the cloud, signalling needs no session affinity (it is one request), but the media must
  reach the process that answered. Use TURN (`relay`) when the server has no public address.

## Limitations

* There is no trickle ICE: the client must gather before offering. This is an aiortc
  limitation.
* There is no renegotiation, ICE restart, reconnection or session resumption. A new offer
  starts a new session.
* There is no server-side echo cancellation. It is not needed, since browsers do it.
* aiortc's Opus encoder runs at a fixed 96 kb/s, and its receive jitter buffer is not
  tunable.
* aiortc uses one STUN and one TURN server per peer connection.
* There is no DTMF (`capabilities.dtmf` is false). Use the telephony transports for phone
  calls.
