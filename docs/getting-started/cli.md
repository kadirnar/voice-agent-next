# The `van` CLI

`van` (also installed as `voice-agent-next`) runs agents, checks the machine, manages
models, serves engines and runs benchmarks. Every command has `--help`.

| Command | What it does |
|---|---|
| `van version` | Print the installed version |
| `van doctor [--network] [--mic] [--echo] [--latency] [--json]` | Diagnose the machine: Python, audio host APIs and devices, GPUs, presets, models; opt-in endpoint, microphone, echo and latency probes ([van doctor](../cli/doctor.md)) |
| `van devices [--json]` | List audio devices: index, host API, channels, default rate, system defaults |
| `van providers [--kind K] [--json]` | List registered providers and whether they are ready here ([providers](../providers/index.md)) |
| `van presets [NAME] [--json]` | List presets and what each one still needs; one preset in detail ([presets](../presets.md)) |
| `van run` | Run a voice agent from a preset, a config file and/or flags |
| `van demo [--turns N]` | Offline demo: a simulated user talks to the mock engine |
| `van models list/download/verify/prune/path/du` | Model manager for local models ([models](../models.md)) |
| `van serve` | Serve engines over the OpenAI Realtime protocol at `/v1/realtime` ([realtime server](../deploy/realtime-server.md)) |
| `van bench latency/asr/overhead/report` | Benchmarks measured on the call recording ([methodology](../benchmarks/methodology.md)) |

## `van run`

```bash
van run                                   # the best preset that is ready on this machine
van run --preset cloud-fast               # a preset (checked first; prints the fixes if not ready)
van run -c agent.yaml                     # a config file (YAML, TOML or JSON)
van run --engine openai/gpt-realtime      # a native speech-to-speech engine
van run --stt deepgram/nova-3 --llm groq --tts cartesia --vad silero --turn smart_turn
van run --transport file --input question.wav --output reply.wav
van run --transport websocket             # serve the browser client instead of local audio
```

| Option | Meaning |
|---|---|
| `-p, --preset NAME` | Named configuration (`van presets` lists them) |
| `-c, --config PATH` | YAML/TOML/JSON config; `extends:` starts it from a preset |
| `--engine SPEC` | Native S2S engine, e.g. `openai/gpt-realtime` |
| `--stt`, `--llm`, `--tts`, `--vad`, `--turn` | Cascade components, e.g. `deepgram/nova-3`, `openai/gpt-4.1-mini`, `cartesia/sonic-2`, `silero`, `smart_turn` |
| `--instructions TEXT`, `--greeting TEXT` | System prompt, spoken greeting |
| `--transport TYPE` | `local` (default), `file`, `websocket`, `webrtc`, `twilio`, `telnyx`, `vonage`, `plivo` |
| `--input PATH`, `--output PATH` | WAV files for the file transport |
| `--skip-checks` | Run a preset without the readiness checks |
| `-v, --verbose` | Debug logging |

Precedence: flags override the config file, which overrides its `extends:` preset. Setting
`--engine` drops the preset's cascade components and vice versa ([presets](../presets.md#extends-in-config-files)).

## Config files

`van run -c`, `van serve -e` and `van bench --config` read the same format:

```yaml
extends: cloud-fast                 # optional: start from a preset
stt: deepgram/flux-general-en       # a spec, or a mapping with provider options:
tts: {provider: cartesia, voice: "..."}
llm: [anthropic/claude-haiku-4-5, groq/gpt-oss-120b]   # a failover chain
agent:
  instructions: You are a helpful assistant.
  greeting: Hello!
turn_detector: smart_turn
session: {allow_interruptions: true}   # SessionOptions
cascade: {preemptive_generation: true} # CascadeOptions
transport: {type: local, echo_mode: headphones}
```

## `van serve`

```bash
van serve --engine mock                                   # OpenAI Realtime protocol on 127.0.0.1:8000
van serve --stt faster_whisper --llm ollama/qwen3 --tts kokoro --vad silero
van serve -e fast=agent-fast.yaml -e smart=agent-smart.yaml --host 0.0.0.0 --api-key "$KEY"
```

Any engine or cascade, behind the OpenAI Realtime protocol, so OpenAI SDKs and Realtime
clients can use it. Options and protocol mapping: [realtime server](../deploy/realtime-server.md).

## `van bench`

```bash
van bench latency --engine mock --turns 20       # T1 voice-to-voice latency (offline smoke run)
van bench latency --config agent.yaml            # ... of the stack `van run -c agent.yaml` runs
van bench asr --stt faster-whisper/base          # T2 WER/CER, RTFx, time to first/final segment
van bench overhead                               # T7 framework overhead (the CI gate)
van bench report bench-results/<run-id>          # re-render report.md
```

See [methodology](../benchmarks/methodology.md) and [results](../benchmarks/results.md).
