# Serving in production: `van serve`

`van serve` runs voice agents as a network service. One command covers every protocol.
It prewarms engines, limits concurrent sessions, runs worker processes, drains gracefully,
and exposes health and Prometheus endpoints.

```bash
van serve -p openai-realtime -e agent.yaml --host 0.0.0.0 --api-key "$KEY" --prewarm 2
van serve -p websocket --preset local-cpu --max-sessions 8 --workers 4
van serve -p twilio --config agent.yaml --host 0.0.0.0 --port 8765 --prewarm 1
van serve -p webrtc --engine openai/gpt-realtime --max-sessions 20
```

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

`--max-sessions N` caps live sessions **per process**. Beyond the cap, a new session is
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
* The telephony protocols serve the media-stream WebSocket only. Your webhook answers the
  call with markup (TwiML and similar) that points at it. See
  `voice_agent_next.transports.telephony.markup`.
* A prewarmed connection is used for one call only. Engines that hold GPU memory per
  connection keep N of those allocated while idle.
* With a shared engine, engine-level usage metrics (`session.usage`) are not attributed to
  a single session. Turn metrics and the voice-to-voice histogram are.
