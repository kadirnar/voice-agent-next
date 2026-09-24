# 09 · Benchmark an agent with `van bench`

`van bench` measures what the caller **hears**. A simulated caller streams pre-recorded
questions in real time, the agent's reply is recorded on the same clock, and the latency
is read off that stereo recording. The session's own metrics are reported next to it and
explain where the time went. Every engine works: native speech-to-speech or a cascade,
local or cloud. The full reference is [benchmarks/README.md](../benchmarks/README.md).

There are two tracks:

| Command | Question it answers |
| --- | --- |
| `van bench latency` (T1) | How long after the user stops talking does the agent start? How often is there dead air? |
| `van bench overhead` (T7) | How much latency does the framework itself add on top of its components? |

## 1. Offline smoke run (mock engine, ~6 s)

```console
$ van bench latency --engine mock --turns 3 --warmup-turns 0 --no-audio --out bench-results/
```

The mock engine ends a turn after 400 ms of silence (energy VAD) and then answers at
once. So voice-to-voice is about **400 ms**, and almost all of it is the endpointing
delay. The tests run this exact command, so it stays correct.

## 2. Benchmark a real agent

Use the same specs and config files as `van run`:

```console
$ van bench latency --engine openai/gpt-realtime-2.1 --turns 20
$ van bench latency --stt sherpa-onnx --llm ollama/qwen3.5:4b --tts kokoro --vad silero --turns 20
$ van bench latency --config agent.yaml --scenario benchmarks/scenarios/latency-conversational.yaml --turns 40 --sessions 3
```

* `--turns` is the number of turns per session, and `--sessions` the number of separate
  sessions. For numbers you publish, measure at least 100 turns over at least 3 sessions.
* The first `--warmup-turns` turns of every session (default 1) are reported as the cold
  start only. They are left out of the headline numbers.
* `--scenario` picks what the caller says. The built-in `latency-smoke` scenario uses
  synthetic speech. Models that need real words (every STT) need `latency-conversational`
  or `latency-tts`, or your own YAML file (see [the scenarios](../benchmarks/scenarios)).
* `--reference-vad` finds where the agent starts speaking in the recording. The default,
  `rms`, is deterministic and does not depend on the engine under test.

## 3. Read the results

The terminal shows a table. The same numbers go to `bench-results/<run-id>/`:

| File | What is in it |
| --- | --- |
| `report.md` | the table, the rates, the method and a per-turn table: start here |
| `summary.json` | every distribution (n, mean, p50/p90/p95/p99, 95 % CI), rates and counts |
| `items.jsonl` | one line per turn: every metric, transcripts, errors |
| `manifest.json` | what ran, where: config (keys redacted), scenario hash, versions, CPU/GPU |
| `artifacts/session-NNN/` | `stereo.wav` (user left, agent right), `labels.txt` for Audacity, `timeline.jsonl` |

The rows, in the order you usually read them:

| Row | Meaning | What to do with it |
| --- | --- | --- |
| **voice-to-voice** `v2v_ms` (recording) | end of user speech → first agent speech, as heard. **p50 is the headline, p90/p95 the tail** | compare systems on this |
| first turn (cold start) | the same for the first turn of each session | a high value means you should warm up models (`warmup()`) |
| session `TurnMetrics` v2v | the session's own estimate | should be close to the recording |
| residual (recording − session) | transport and playout buffering, speech-end estimation error | ≈ 0 on the loopback transport. If it is large, the session's metric is misleading |
| end-of-turn delay | user stopped → turn committed (VAD silence, turn detector, STT final) | usually the largest part: tune `min_endpointing_delay`, VAD silence, the turn detector |
| response TTFB | turn committed → first audio (LLM TTFT + TTS TTFB, or the S2S model) | provider and model choice |
| `stt` / `llm` TTFT / `tts` TTFB | per-component first-result latency (cascades) | find the slow component |
| dead air | share of turns over 2 s (`--dead-air`) or without a reply | should be 0 % |
| missed / premature / interrupted | no reply / agent started before the user finished / reply cut off | premature > 0 means endpointing is too eager |

In short: **v2v ≈ end-of-turn delay + response TTFB + residual**. Improve the largest term
first.

Re-render a report, or load a run in Python:

```console
$ van bench report bench-results/<run-id>
```

```python
from voice_agent_next.bench import load_run

run = load_run("bench-results/<run-id>")
print(run.summary.metrics["v2v_ms"].p50, run.summary.rates)
```

## 4. Framework overhead and the regression gate

```console
$ van bench overhead                                     # smoke tier, ~2 min
$ van bench overhead --sections micro                    # hot-path micro-benchmarks only, ~10 s
$ van bench overhead --baseline benchmarks/baselines/overhead-ci.json --retries 1
```

`overhead` runs mock systems whose delays are known (`injected_ms`). Whatever latency is
left over is the runtime's own cost: `overhead_ms = v2v_ms − injected_ms`. On Linux it is
about 2 ms. The other rows are frame jitter, event-loop lag, how fast playback stops on an
interrupt (`flush_ms`, and leaked frames), sessions per CPU core, and micro-benchmarks in
µs per operation. With `--baseline`, the command exits with status 1 when a metric
regressed compared with the stored entry for this OS. This is the CI gate.

Do not commit `bench-results/`: it is git-ignored.
