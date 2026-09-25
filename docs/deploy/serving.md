# Serving in production: `van serve`

`van serve` runs voice agents as a network service. One command covers every protocol.
It prewarms engines, limits concurrent sessions, runs worker processes, drains gracefully,
and exposes health and Prometheus endpoints.

```bash
van serve -p openai-realtime -e agent.yaml --host 0.0.0.0 --api-key "$KEY" --prewarm 2
van serve -p websocket --preset local-cpu --max-sessions 8 --workers 4
van serve -p twilio --config agent.yaml --host 0.0.0.0 --port 8765 --prewarm 1
van serve -p webrtc --engine openai/gpt-realtime --max-sessions 20
van serve -p websocket --preset local-cpu --allowed-origin https://app.example.com
```

The defaults are secure: loopback only, browser pages from other websites refused, and
limits on sessions, their duration and idle time. See [Secure defaults](#secure-defaults).

## Protocols

| `--protocol` | Server | Clients connect to | Refusal (busy or draining) |
| --- | --- | --- | --- |
| `openai-realtime` (default) | `RealtimeServer` ([details](realtime-server.md)) | `ws://HOST:8000/v1/realtime` | HTTP 503 + `Retry-After: 1`, OpenAI-style `error` body (`session_limit_reached`, `server_draining`) |
| `websocket` | `WebSocketAgentServer` ([van-ws/1](../transports/websocket.md)) | `ws://HOST:8765/` | HTTP 503 + `Retry-After: 1`, body `{"type": "error", "code": "server_busy" \| "server_draining"}` |
| `webrtc` | `WebRTCAgentServer` ([WebRTC](../transports/webrtc.md)) | `POST http://HOST:8080/offer` | HTTP 503 `{"error": ...}` to the offer |
| `twilio`, `telnyx`, `vonage`, `plivo` | `TelephonyServer` ([telephony](../transports/telephony.md)) | `ws(s)://HOST:8765/` media stream | HTTP 503 on the media-stream upgrade, so the provider fails the stream |

`openai-realtime` serves engines: clients bring their own instructions and tools. The other
protocols run a full `AgentSession` per connection, with barge-in, tools, the agent's
greeting and metrics. You build that agent from one of these sources:

* `--preset NAME`: a preset (`van presets` lists them).
* `--config agent.yaml`: an agent config with the engine or cascade, `agent:` and `session:`.
* `--engine SPEC`: a native engine, for example `openai/gpt-realtime` or `mock`.
* cascade flags: `--stt`, `--llm`, `--tts`, `--vad` and `--turn`.

`--instructions`, `--voice` and `--language` override the agent's values. With no source,
`van serve` serves the offline mock engine.

## Secure defaults

`van serve` and the server classes behind it are safe to start without extra flags. On a
laptop they are reachable only from the machine itself, and the limits below always apply.

### Who may connect: the Origin allow-list

Browsers send an `Origin` header with every WebSocket upgrade. Without a check, any website
a user visits could open a session from their browser to an agent on their machine or LAN
and spend its engine (cross-site WebSocket hijacking). The WebSocket servers
(`openai-realtime`, `websocket` and the telephony protocols) therefore accept:

* clients that send **no `Origin`**: native clients, SDKs, backends and telephony providers;
* pages served **from this machine**: `http(s)://localhost:*`, `127.0.0.0/8`, `[::1]` and
  `*.localhost`, on any port;
* the origins you list with `--allowed-origin` (repeatable, or `VAN_ALLOWED_ORIGINS`
  separated by spaces).

Any other page gets HTTP 403 with the code `origin_not_allowed`, and the server logs the
refused origin.

```bash
van serve -p websocket --allowed-origin https://app.example.com
van serve -p websocket --allowed-origin 'https://*.example.com'   # every subdomain
van serve -p websocket --allowed-origin http://intranet:8080      # a non-default port
```

`null` (sandboxed iframes and `file://` pages) must be listed explicitly. `*` allows every
origin; use it only when an authenticating proxy protects the server. For `webrtc`,
`--allowed-origin` sets the CORS origins of the signalling endpoints.

### Listening beyond this machine

The default `--host` is `127.0.0.1`. To accept remote clients, bind `0.0.0.0` (or an
interface address):

* **`openai-realtime` refuses to start without `--api-key`** (or `VAN_SERVER_API_KEY`). The
  protocol has bearer-token authentication, so an open endpoint is almost always a mistake
  that lets anyone use (and pay for) the engines.
* The **other protocols print a warning**. They have no built-in authentication:
  telephony providers and WebRTC peers cannot send a key. Authenticate in front of them
  (a reverse proxy, a `process_request` hook, or a session factory that raises
  `SessionRefused`), and restrict the network.
* `--insecure` allows the unauthenticated bind and silences the warning, for servers that
  a firewall, a private network or an authenticating proxy already protects.

Containers bind `0.0.0.0` inside the container: publish the port on `127.0.0.1` or behind
your proxy, and pass `--api-key` for `openai-realtime` ([Docker](docker.md#running)).

### Session limits

| Flag | Default | What happens |
| --- | --- | --- |
| `--max-sessions N` | 64 per process | New sessions get HTTP 503 (`Retry-After: 1`) |
| `--max-session-duration S` | 3600 s | The session gets an `error` with `session_expired` and is closed (1000) |
| `--idle-timeout S` | 300 s | A session that received no client message for `S` seconds gets `session_idle` and is closed (1000) |

`0` disables a limit. Clients that stream microphone audio are never idle. Telephony calls
stream audio continuously, too. The duration and idle limits apply to `openai-realtime`,
`websocket` and the telephony protocols. For `webrtc`, `--max-sessions` applies.

In Python, the same options are `max_sessions`, `max_session_duration`, `idle_timeout` and
`allowed_origins` of `RealtimeServer` and `WebSocketAgentServer` (`None` disables a
limit).

### Bounded queues

Neither direction of a WebSocket connection can grow the server's memory without bound:

* **A client that sends faster than the session consumes** (a flood of audio): above
  4 MiB of queued input, the server stops reading the socket until the session is back
  under 1 MiB. TCP backpressure then slows the client down, and no audio is dropped.
* **A client that does not read** what it is sent: above `max_send_buffer` (32 MiB) of
  queued output, the connection is closed with 1008 and the queue is freed. The Realtime
  server also stops taking engine output above 2 MiB, so a slow reader only pauses the
  engine.
* **Typed messages** (`text`) waiting for a reply are capped at 8 per `websocket`
  connection. Beyond that, the client gets `rate_limited`.

### Error messages

Clients never see exception text, which can contain file paths, hosts, credentials in
URLs or provider responses. They get a generic message with a correlation id:

```json
{"type": "error", "code": "internal_error", "fatal": true, "error_id": "err_3f9a1c0b7d2e",
 "message": "The session failed (error id err_3f9a1c0b7d2e)."}
```

The server logs the full error (with the traceback for unexpected failures) under the same
id. Search the logs for the id a user reports. With `--log-format json`, the log record also
has an `error_id` field. A session factory that raises
`voice_agent_next.errors.SessionRefused("...")` refuses the client with exactly that
message, for example after failed authentication.

## Prewarm: no model load or connection setup in the call path

Before a new call can play audio, two things must happen:

1. **Loading models.** `engine.warmup()` loads VAD, STT and TTS weights, or an ONNX
   session. This takes seconds, and it happens once per process.
2. **Setting up the connection.** `engine.connect()` opens a WebSocket to a cloud API, or
   the per-connection VAD and STT state of a cascade.

`van serve` loads the models when the process starts. `--prewarm N` then keeps **N engine
connections open and ready**. A new call takes one, and the pool refills in the background.
A prewarmed connection is opened with the options the call will use: the agent's
instructions, tools, voice and language. For `openai-realtime` those are the session
defaults.

* If the call connects with the same options, it gets the prewarmed connection as is.
* If only the instructions or tools differ (typical for Realtime clients that send
  `session.update`), the connection is updated in place. This works only on engines that
  implement `EngineConnection.update`.
* A different voice or language, or a call that starts with history, gets a fresh
  connection. The unused prewarmed connection is closed.

Prewarmed connections older than `--prewarm-max-idle` seconds (default 300) are replaced
with new ones, because cloud sessions and proxies drop idle sockets. Keep `--prewarm` small
for cloud engines: some providers bill or rate-limit open sessions.

`--engine-per-session` gives every call **its own engine instance**. The default is one
engine per process with one connection per call. Use it for engines that are not safe to
share across concurrent calls. The pool then builds and warms up N whole engines ahead of
time. Each engine is used by one call and closed when that call ends.

### Measured time to first audio

Setup: a new `websocket` call opens and the agent speaks its greeting. The time runs from
the client's connect to the first agent audio byte. The engine is a local cascade on CPU
(Silero VAD, Kokoro ONNX TTS, mock STT and LLM) on a 12-thread Linux x86-64 machine, with
other jobs running (load average about 5). There were two runs of five consecutive calls
each, and every mode ran in a fresh process.

| Server | 1st call | Calls 2–5 |
| --- | --- | --- |
| `--no-warmup` (cold) | 1290 / 1264 ms | 384–587 ms |
| default (models warmed at start) | 334 / 451 ms | 328–509 ms |
| `--prewarm 1` | 520 / 339 ms | 370–908 ms |

On a local cascade, most of the cold start is **model loading**, about 0.8–0.9 s here. The
default warm-up removes it from the first call. After that, time to first audio is mostly
Kokoro synthesizing the greeting (0.3–0.5 s on a busy CPU). Opening a cascade connection
is cheap, so `--prewarm` changes nothing measurable for this engine.

`--prewarm` pays off when **connection setup** is slow, as with cloud speech-to-speech
APIs, which set up a WebSocket session with TLS for each call. With a simulated 300 ms
connection setup, a pool-level measurement gives the time from lease to a connected
engine:

| Pool | Calls 1–3 |
| --- | --- |
| `--prewarm 0` | 300.5, 300.5, 300.4 ms |
| `--prewarm 1` | 0.0, 0.1, 0.1 ms |

In production, `van_engine_connect_seconds{prewarmed="true"|"false"}` shows the same split.

## Concurrency limits and admission

`--max-sessions N` caps live sessions **per process** (default 64, `0`: no limit). Beyond the cap, a new session is
refused before the WebSocket upgrade (or before the WebRTC offer is answered) with HTTP
503 and `Retry-After: 1`. A load balancer or client can then retry on another instance.
Refusals are counted in `van_sessions_rejected_total{reason="busy"}`.

Size the cap from a benchmark, not a guess. `van bench` has an overhead track that measures
sessions per core and the RSS and CPU per session. Also count the RAM of each engine
instance when you use `--engine-per-session`.

## Worker processes

`--workers N` starts a supervisor and N worker processes. Each worker loads its own models
and prewarms its own pool. All workers bind the same port with `SO_REUSEPORT`, so the
kernel spreads new connections across them. A worker that crashes is restarted. A worker
that fails at startup with a configuration error stops the service.

* **Linux**: the kernel balances connections across the workers.
* **macOS**: works, but the BSD `SO_REUSEPORT` does not balance evenly. Prefer one process
  per instance behind a load balancer.
* **Windows**: there is no `SO_REUSEPORT`. `van serve` warns and runs **one process**. For
  more capacity, run several instances on different ports behind a load balancer.

Each worker serves its own `/health`, `/ready` and `/metrics`. With `SO_REUSEPORT`, a
scrape reaches one random worker, and the `worker="N"` label shows which. To see every
worker, run one instance per container and scrape each one.

Workers are separate processes, so CPU-heavy VAD or resampling in one of them cannot slow
calls in another. Inside a worker, sessions share one event loop. Blocking inference runs
in threads.

## Graceful drain

The first SIGTERM or Ctrl+C starts a **drain**:

1. `/ready` returns 503 with `"reasons": ["draining"]`, and new sessions are refused (503).
2. Live calls continue until they end, for up to `--drain-timeout` seconds (default 30).
3. Calls still running after that are closed (WebSocket 1001). Then the pools and engines
   are closed, and the process exits.

A second signal skips the wait. With workers, the supervisor forwards the signal to every
worker. On Kubernetes, set `terminationGracePeriodSeconds` above `--drain-timeout`.

## Health, readiness and metrics

Every protocol serves these routes on its own port:

| Route | Meaning |
| --- | --- |
| `GET /health` (`/healthz`, `/livez`) | Liveness: 200 while the process serves, draining included. The body has the status, protocol, pid, worker, session count and uptime. |
| `GET /ready` (`/readyz`) | Readiness: 200 when a new call would be served warm now. That needs the models loaded, a prewarmed engine available (with `--prewarm`), room under `--max-sessions`, and no drain. Otherwise 503 with `reasons`. |
| `GET /metrics` | Prometheus text format (below). |

Point the liveness probe at `/health` and the readiness probe at `/ready`. `/ready` also
turns 503 while the pool refills after a burst of calls, so the load balancer sends new
calls to warm instances.

| Metric | Type | Meaning |
| --- | --- | --- |
| `van_up`, `van_ready`, `van_draining` | gauge | Process state |
| `van_uptime_seconds` | gauge | Time since the process started |
| `van_sessions_active` / `van_sessions_max` | gauge | Live sessions and the limit (0: none) |
| `van_sessions_total` | counter | Sessions started |
| `van_sessions_rejected_total{reason}` | counter | Refused sessions (`busy`, `draining`) |
| `van_session_errors_total` | counter | Sessions that ended with an error |
| `van_turns_total` | counter | Agent turns completed (`TurnMetrics`) |
| `van_pool_size`, `van_pool_idle`, `van_pool_leased` `{pool}` | gauge | Prewarm target, ready items, engines in use |
| `van_pool_hits_total`, `van_pool_misses_total`, `van_pool_errors_total` `{pool}` | counter | Warm and cold starts, failed prewarms |
| `van_pool_warmup_seconds{pool}` | gauge | Duration of the initial warm-up |
| `van_voice_to_voice_seconds` | histogram | User stops speaking → first agent audio, from `TurnMetrics` |
| `van_engine_connect_seconds{prewarmed}` | histogram | Time to a session's first engine connection |

Every sample carries `protocol` and, with workers, `worker`. The voice-to-voice histogram
comes from the `AgentSession` protocols. For `openai-realtime`, the client plays the audio,
so measure voice-to-voice on the client.

## Logs

`--log-format json` writes one JSON object per line: `ts`, `level`, `logger`, `msg`,
`pid`, `worker` and `session_id`, plus the record's extra fields. `--log-format text` (the
default) appends `[session=...]`. Every log line from inside a session carries its session
id, including lines from the engine's tasks. The ids are `sess_...` for Realtime sessions,
`ws_...` for the WebSocket and telephony protocols, and `rtc_...` for WebRTC.

## From Python

```python
import asyncio

from voice_agent_next.config import load_config
from voice_agent_next.server.serving import build_agent_served, run_served

served = build_agent_served(
    "websocket", load_config("agent.yaml"), prewarm=2, max_sessions=10, port=8765
)
asyncio.run(run_served(served, drain_timeout=30))
```

`build_realtime_served(models, ...)` does the same for the OpenAI Realtime protocol.
`EnginePool` (`voice_agent_next.server.pool`) and `ServeState`
(`voice_agent_next.server.ops`) are the building blocks, if you mount the protocol servers
in your own application.

## Limitations

* `--api-key` applies to `openai-realtime` only. Put the other protocols behind an
  authenticating reverse proxy, and use TLS (`wss://`) in production.
* `webrtc` has no session duration or idle limit yet: `--max-session-duration` and
  `--idle-timeout` do not apply to it.
* The telephony protocols serve the media-stream WebSocket only. Your webhook answers the
  call with markup (TwiML and similar) that points at it. See
  `voice_agent_next.transports.telephony.markup`.
* A prewarmed connection is used for one call only. Engines that hold GPU memory per
  connection keep N of those allocated while idle.
* With a shared engine, engine-level usage metrics (`session.usage`) are not attributed to
  a single session. Turn metrics and the voice-to-voice histogram are.
