# Presets

A preset is a named, tested agent configuration: the engine or the cascade components,
plus what they need to run (extras, API keys, platform, GPU, local model server).
`voice_agent_next.presets` holds them; `van presets` tells you which ones run on your
machine and how to fix the others.

```console
$ van presets                  # every preset, ready or what is missing
$ van presets local-cpu        # one preset: why these components, its config, the fixes
$ van run --preset local-cpu   # check, then run (prints the fixes and exits 1 if not ready)
$ van run                      # no preset/config/flags: the first ready preset, named
```

## The presets

| Preset | Where | Stack | Needs |
|---|---|---|---|
| `local-gpu` | local | faster-whisper `large-v3-turbo` (CUDA) · Ollama `qwen3.5:9b` · Kokoro v1.0 fp16 · Silero · Smart Turn | Linux/Windows, an NVIDIA GPU usable by CTranslate2, Ollama |
| `apple` | local | MLX: Parakeet TDT 0.6B v3 (parakeet-mlx) · mlx-lm `Qwen3.5-4B-4bit` → Ollama `qwen3.5:4b` · Kokoro (mlx-audio) · Silero · Smart Turn | Apple silicon with an arm64 Python, the `mlx` extra, `mlx_lm.server` or Ollama |
| `local-cpu` | local | sherpa-onnx `zipformer-en-kroko` · Ollama `LiquidAI/lfm2.5-1.2b-instruct` · Kokoro v1.0 fp16 · Silero · Smart Turn | any OS, Ollama |
| `hybrid` | hybrid | local-cpu's STT/TTS · Claude Haiku 4.5 → Groq `gpt-oss-120b` → local Ollama | one LLM: `ANTHROPIC_API_KEY`, `GROQ_API_KEY` or Ollama |
| `cloud-fast` | cloud | Deepgram Flux · Groq → Cerebras `gpt-oss-120b` · Cartesia Sonic 3.6 | `DEEPGRAM_API_KEY`, `GROQ_API_KEY` or `CEREBRAS_API_KEY`, `CARTESIA_API_KEY` |
| `cloud-quality` | cloud | AssemblyAI Universal-3.5 Pro · Claude Haiku 4.5 → GPT-4.1 · Cartesia → ElevenLabs Flash v2.5 | `ASSEMBLYAI_API_KEY`, an LLM key, a TTS key |
| `openai-realtime` | cloud | OpenAI Realtime `gpt-realtime-2.1` (native speech-to-speech) | `OPENAI_API_KEY` |
| `gemini-live` | cloud | Gemini Live `gemini-3.8-live` (native speech-to-speech) | `GOOGLE_API_KEY` or `GEMINI_API_KEY` |

`→` is a failover chain ([concepts/failover.md](concepts/failover.md)). The local
microphone/speaker transport also needs the `audio` extra. Extras per preset:
`van presets <name>` or `van presets --json`.

### Why these defaults

The choices come from our measurements where we have them and from research note 03 §9.2
([03-stt-tts-llm-landscape.md](research/03-stt-tts-llm-landscape.md)) otherwise. Each
preset's `rationale` (shown by `van presets <name>`) has the details.

* **local-cpu.** On a Ryzen 5 5600 with no GPU, the streaming Kroko Zipformer
  (57 MB) gives the final transcript about 100 ms after speech ends. That cuts the
  end-of-turn delay from 633 ms (faster-whisper `base`) to 400 ms. It also had the best
  voice-to-voice latency: p50 1,126 ms, p90 1,620 ms, 4 % dead air
  ([sherpa-onnx.md](providers/sherpa-onnx.md#latency)). LFM2.5 1.2B is the LLM of every T1
  run, at about 12 ms warm TTFT. It leaves the CPU to STT and TTS; note 03 names CPU
  contention as the main risk of this profile. For better tool calling, use
  `--llm ollama/qwen3.5:4b` (note 03 §7.5). Kokoro v1.0 fp16 is the measured TTS: the
  first clause takes about 400 ms on this CPU.
* **Turn-taking of the local cascades** (`local-cpu`, `local-gpu`, `apple`; `hybrid`
  without preemptive generation). The T4 battery (#113) found 100 % false barge-ins on
  "uh-huh" and 29 % dead air with the library defaults. The presets set
  `cascade: {min_endpointing_delay: 0.5, max_endpointing_delay: 1.5,
  preemptive_generation: true, preemptive_tts: true}` and
  `session: {max_backchannel_duration: 1.0}`: no 2.5 s waits when Smart Turn misjudges a
  short answer, the reply prepared during the endpointing silence, and short utterances
  over the agent treated as backchannels unless they contain an interruption word
  ([endpointing](concepts/endpointing.md#the-local-presets-issue-113),
  [interruptions](concepts/interruptions.md#short-utterances-small-asr-models)). Override
  any of them in your config (`cascade: {min_endpointing_delay: 1.0}` for users who pause
  between sentences).
* **local-gpu.** On an RTX 5070 Ti with CUDA float16, faster-whisper's final transcript
  drops from 391 ms to 51 ms, and v2v p50 drops from 1,523 ms to 969 ms
  ([hardware.md](hardware.md#measurements)). `large-v3-turbo` covers 99 languages at that
  speed. Qwen3.5-9B is note 03's LLM for a 16 GB GPU. The preset fails its check when
  faster-whisper would fall back to CPU (for example, cuBLAS 12 is missing), and the
  check prints the install command.
* **apple.** Everything on the GPU through MLX ([mlx.md](providers/mlx.md)): Parakeet TDT
  0.6B v3 streams through parakeet-mlx (25 European languages) and has the final transcript
  when the turn ends, `mlx_lm.server` runs Qwen3.5-4B at 4-bit (Ollama is the failover when
  no mlx-lm server runs), and Kokoro-82M runs through mlx-audio. The check says how to
  start the mlx-lm server.
* **cloud-fast.** Deepgram Flux ends turns itself, so there is no VAD or endpointing
  delay. gpt-oss-120b on Groq has a 98 / 217 ms p50 / p95 TTFAT in Pipecat's benchmark,
  and Cerebras is the fastest host of the same model. Sonic 3.6 tops the TTS arena.
* **cloud-quality.** Claude Haiku 4.5 has the best tool-use pass rate within 700 ms
  (98.0 % at 637 ms). AssemblyAI U3.5 Pro gives formatted finals with neural end of turn.
* **hybrid.** The audio stays on the machine and only text goes to the cloud. The local
  Ollama model keeps the agent working offline.

## Readiness

`van presets` and `van run --preset` check, for every component:

| Check | Fix printed |
|---|---|
| Python modules of the provider (`requires`) | one `pip install 'voice-agent-next[a,b,c]'` with every missing extra |
| credentials (`env`), unless the spec passes `api_key:` | `set DEEPGRAM_API_KEY (e.g. export DEEPGRAM_API_KEY=...)` |
| platform (the preset's and each provider's) | none: pick another preset |
| `accelerator: cuda`: an NVIDIA GPU, and that faster-whisper's `device="auto"` would use it (`hardware.select_ctranslate2_backend`) | the fix from `van doctor`, e.g. `pip install 'voice-agent-next[cuda]'` |
| `accelerator: apple`: Apple silicon, not an x86_64 Python under Rosetta | install an arm64 Python |
| Ollama: the server answers `GET /api/tags` at the URL the provider uses (`base_url`, `OLLAMA_BASE_URL`, `OLLAMA_HOST`), and has the model (`:latest` implied) | `ollama serve` / `ollama pull <model>` |
| mlx-lm: the server answers `GET /v1/models` at the URL the provider uses (`base_url`, `MLX_LM_BASE_URL`); it loads the model by itself | `python -m mlx_lm.server --model <model>` |
| `--transport local`: `sounddevice` | the `audio` extra |

**Failover lists are pruned.** A chain is ready when at least one member is ready. The
config that runs keeps only the ready members, and a note says which ones were skipped.
For example, `cloud-fast` with only `CEREBRAS_API_KEY` set runs on Cerebras.
`--skip-checks` runs a preset without the checks, for setups the checks cannot see (a
custom `base_url`, a provider from a plugin).

The checks do not download anything or load models. Model downloads happen on first use,
or ahead of time with `warmup()`: Kroko 57 MB, Kokoro fp16 164 MB + 28 MB of voices, Silero 2 MB, Smart
Turn 9 MB, faster-whisper `large-v3-turbo` about 1.6 GB.

## `van run`

What runs, in order of precedence (later wins):

1. `--preset NAME`, or the `extends:` of the config file;
2. the config file (`--config`);
3. component flags (`--engine`, `--stt`, `--llm`, `--tts`, `--vad`, `--turn`),
   `--instructions`, `--greeting`, `--transport`.

The layers are read raw, merged, and validated once at the end, so the config file may hold
only tweaks that make sense on top of the preset (`van run --preset local-cpu -c
tweaks.yaml` with `stt: {language: en}` in `tweaks.yaml`). `van bench` (`--preset` and
`--config`) and `van serve` (`--preset` and `--config`) layer them the same way; in Python,
`config.layer_config(preset=..., file=..., overrides=...)` returns the merged mapping.

The readiness check runs on the final config with the preset's platform and accelerator,
so an override that is not ready fails too (e.g. `--llm ollama/qwen3.5:27b` before
`ollama pull`). Config files and flags without a preset are not checked, as before.

With **no preset, config file or component flag**, `van run` checks the presets in this
order and runs the first ready one: `local-gpu`, `apple`, `local-cpu`, `hybrid`,
`cloud-fast`, `cloud-quality`, `openai-realtime`, `gemini-live`. It prints what it chose.
Local presets come first because they are private and free. If no preset is ready, it
falls back to the mock engine and says so.

```console
$ van run --preset local-cpu --llm ollama/qwen3.5:4b --instructions "You are a barista."
$ van run --preset cloud-fast --transport websocket
$ van run --preset local-cpu --transport file --input question.wav --output reply.wav
```

## `extends:` in config files

A config file can start from a preset and change only what differs:

```yaml
extends: local-cpu
llm: ollama/qwen3.5:4b            # a spec replaces the preset's component
stt: {language: en}               # a mapping without `provider:` only changes options
vad: null                         # null removes a component
agent:
  instructions: You are a friendly barista.
  greeting: Hi! What can I get you?
cascade: {min_endpointing_delay: 0.3}   # sections merge key by key
```

Rules (`config.merge_config`):

* Sections (`agent`, `session`, `cascade`, `transport`) merge key by key. A `transport`
  with a different `type` replaces the base one.
* Setting `engine:` drops the preset's cascade components, and setting a cascade
  component drops the preset's `engine:`.
* An options-only mapping cannot be applied to a failover list. Give the whole list.

`extends:` takes a preset name; `AppConfig.extends` records it. `${ENV_VAR}` strings are
expanded before `extends:` is read, so `extends: ${VAN_PRESET:-local-cpu}` picks the preset
from the environment.

## Python

```python
from voice_agent_next import build_agent, build_session, load_preset, session_from_preset
from voice_agent_next.presets import check_preset

cfg = load_preset("cloud-fast", agent={"instructions": "You are a travel agent."})
session, agent = build_session(cfg), build_agent(cfg)

# or in one step
session, agent = session_from_preset("local-cpu", llm="ollama/qwen3.5:4b")

# readiness, without raising
result = check_preset("local-gpu")
print(result.ready, result.fixes(), result.notes)
```

`load_preset(name, *, check=True, env=None, **overrides)` merges `overrides` like
`extends:` does. It raises `ConfigurationError` with the list of fixes when the preset
cannot run, and prunes failover lists. `check=False` returns the preset config as is.
`check_preset` / `check_config` take an `Environment` whose probes can be replaced:
platform, environment variables, installed modules, GPUs, Apple silicon, the CTranslate2
backend and Ollama's model list. The tests use this to fake a machine.
