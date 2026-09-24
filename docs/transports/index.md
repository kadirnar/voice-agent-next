# Transports

A transport moves audio between the user and the `AgentSession`: user audio in, agent
audio out, plus optional pause/resume, playback position, DTMF and a data channel. The
session resamples to whatever format the transport uses and paces playback itself, so
engines and transports combine freely.

```python
from voice_agent_next.transports import create_transport

await session.run(agent, create_transport("local"))  # microphone + speakers
await session.run(
    agent, create_transport({"type": "file", "input_path": "q.wav", "output_path": "a.wav"})
)
```

| Transport | `type` | Extra | Use it for | Page |
|---|---|---|---|---|
| Local audio | `local` | `audio` | microphone and speakers on this computer, with echo cancellation | [Local audio](local.md) |
| WebSocket (`van-ws/1`) | `websocket` | — | browsers on good networks, server-to-server, LAN and localhost apps | [WebSocket](websocket.md) |
| WebRTC (`van-webrtc/1`) | `webrtc` | `webrtc` | browsers and mobile apps across the internet (Opus, jitter buffer, browser AEC) | [WebRTC](webrtc.md) |
| Telephony | `twilio`, `telnyx`, `vonage`, `plivo` | — | phone calls through a provider's media streams | [Telephony](telephony.md) |
| File | `file` | — | a WAV file in, the reply to a WAV file (batch tests, demos) | [API](../reference/transports.md) |
| Loopback | `loopback` | — | in-memory: tests, simulated callers, benchmarks | [API](../reference/transports.md) |

`van run --transport <type>` selects one from the command line; in a config file it is the
`transport:` section.

## Capabilities

`TransportCapabilities` tells the session what a transport can do:

| Capability | Effect |
|---|---|
| `pause` | playback can be paused and resumed, so a false barge-in (a cough, "uh-huh") resumes where it stopped ([interruptions](../concepts/interruptions.md#pausing-and-transports)) |
| `playback_position` | `buffered_duration()` reflects what the listener actually heard (device latency, browser playback reports, telephony marks); truncation uses it |
| `dtmf` | emits `"dtmf"` events with the pressed key |
| `messages` | a JSON data channel: transcripts, state, metrics, typed input |

## Writing a transport

Subclass `Transport` (input/output `AudioFormat`, `audio_input()`, `write_audio()`,
`clear_audio()`, optional `pause_audio()` / `resume_audio()`, `buffered_duration()` and
`send_message()`),
declare its capabilities, and add it to `create_transport` ([API](../reference/transports.md)).
Deployment guides: [deploy](../deploy/index.md).
