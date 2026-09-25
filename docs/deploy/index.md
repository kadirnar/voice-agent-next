# Deploy

How users reach your agent in production:

| You want | Use | Guide |
|---|---|---|
| Existing OpenAI Realtime clients (OpenAI SDK, Agents SDK, LiveKit/Pipecat plugins) to talk to any engine, including a fully local cascade | `van serve` | [Realtime server](realtime-server.md) |
| A web or mobile app over the internet | WebRTC transport + one HTTP signalling endpoint | [WebRTC](../transports/webrtc.md) |
| A web app on a good network, or a backend-to-backend link | WebSocket transport | [WebSocket](../transports/websocket.md) |
| Phone calls | Twilio, Telnyx, Vonage or Plivo media streams | [Telephony](../transports/telephony.md) |
| Containers: a fully local stack with `docker compose up`, or images for your cluster | Docker images (CPU, CUDA) and compose files | [Docker](docker.md) |

## One session per connection

Every server transport (`serve_websocket`, the WebRTC server, the telephony server, `van serve`)
creates one `AgentSession` per connection or call from a factory, so sessions share
nothing but what you give them. Share heavy objects between sessions explicitly: create a
local model once (`create("stt", ...)`) and pass the instance, and call `engine.warmup()`
before the first caller arrives (`van serve` warms engines up by default).

## Checklist

* **Latency.** Put the agent close to the users and the model providers; measure the
  stack you deploy with `van bench latency` ([methodology](../benchmarks/methodology.md)).
* **Failover.** Give `stt`, `llm` and `tts` a fallback chain ([failover](../concepts/failover.md)).
* **Observability.** Record calls (stereo WAV + JSONL timeline) and trace with
  OpenTelemetry ([observability](../concepts/observability.md)).
* **Authentication.** `van serve --api-key` (or `VAN_SERVER_API_KEY`). The WebSocket server
  can authenticate before the upgrade (`process_request`) or in the session factory from
  the client's hello metadata; mount the WebRTC offer handler behind your own auth (see
  their pages).
* **Hardware.** Local models pick a GPU automatically; check with `van doctor`
  ([hardware](../hardware.md)). Pre-download models into images with `van models download`
  ([models](../models.md#docker-images-and-offline-machines)).
* **Session limits.** Native engines with a provider session limit rotate transparently
  (`EngineCapabilities.max_session_duration`); `van serve --max-session-duration` caps
  sessions on your side.
