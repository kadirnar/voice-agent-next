# Examples

Small, runnable scripts for the common scenarios. Each one is a complete program: read it
top to bottom, copy what you need.

**Every example runs offline with `--mock`.** The mock engine and providers, or a local
fake of the cloud API, replace the models. A WAV file or an in-memory transport replaces
the microphone and speakers. No API keys, downloads or audio devices are needed.
`tests/test_examples.py` runs all of them this way, so they stay in sync with the library.

```console
$ uv sync                                   # or: pip install voice-agent-next
$ python examples/01_offline_local_agent.py --mock
user : What can you do offline?
  (voice-to-voice latency 4 ms)
agent: I can chat, call tools and keep your audio private.
(session closed: user_disconnected)
```

Without `--mock`, the examples use real providers and, unless you pass `--wav`, your
microphone and speakers. Each script's docstring lists the extras, keys and commands it
needs.

| # | Example | Shows | Needs without `--mock` |
|---|---|---|---|
| 01 | [`01_offline_local_agent.py`](01_offline_local_agent.py) | the `local-cpu` preset over mic and speakers, `--wav` for a file, readiness checks | Ollama, extras `audio,sherpa-onnx,openai,kokoro,silero,smart-turn` |
| 02 | [`02_openai_realtime.py`](02_openai_realtime.py) | OpenAI Realtime speech-to-speech with a tool. `--mock` runs the real client against a local server that speaks the same protocol | `OPENAI_API_KEY` |
| 03 | [`03_gemini_live.py`](03_gemini_live.py) | Gemini Live with a tool. `--mock` uses `FakeGeminiLiveServer` | `GOOGLE_API_KEY` |
| 04 | [`04_cascade_mix.py`](04_cascade_mix.py) | cloud STT + local LLM + TTS with failover chains, `--dry-run` readiness report | Deepgram/Cartesia keys (optional), Ollama |
| 05 | [`05_tools.py`](05_tools.py) | blocking and non-blocking tools, fillers, progress updates | any engine (`--engine`) |
| 06 | [`06_telephony_twilio.py`](06_telephony_twilio.py) | Twilio Media Streams server with a TwiML webhook. `--mock` places a simulated call | a Twilio number, a public URL (ngrok) |
| 07 | [`07_realtime_server.py`](07_realtime_server.py) | `van serve`: any engine behind the OpenAI Realtime API, used from the `openai` SDK | `pip install 'voice-agent-next[openai]'` for the SDK client |
| 08 | [`08_recording_and_tracing.py`](08_recording_and_tracing.py) | stereo call recording + JSONL timeline, OpenTelemetry spans | extra `otel` + `opentelemetry-sdk` for spans |
| 09 | [`09_benchmark.md`](09_benchmark.md) | `van bench latency` / `van bench overhead`, and how to read the results | — |
| 10 | [`10_custom_provider.py`](10_custom_provider.py) | writing a provider, `@register_provider`, plugin entry points | — |

Other demos:

* [`websocket_agent.py`](websocket_agent.py) and [`web/`](web): an agent over WebSocket
  (`van-ws/1`) with a browser page and a Python client.
* [`webrtc/`](webrtc): an agent over WebRTC (aiortc) with a browser demo.

[`_common.py`](_common.py) has the few helpers the examples share: printing the
conversation, writing a synthetic test WAV, and a simulated caller.

## Mock mode in your own tests

The same approach tests your agent offline:

* `voice_agent_next.providers.mock`: `MockEngine`, `MockSTT`, `MockLLM` and `MockTTS`, with
  scripted transcripts, replies and tool calls (`MockToolCall`), and configurable
  latencies;
* `voice_agent_next.testing.gemini_live.FakeGeminiLiveServer`: the Gemini Live protocol
  on localhost;
* `voice_agent_next.server.RealtimeServer(MockEngine(...))`: an OpenAI Realtime endpoint
  on localhost, for clients such as `OpenAIRealtimeEngine` or the `openai` SDK;
* `FileTransport(wav, realtime=False)` or `LoopbackTransport` in place of audio devices.
