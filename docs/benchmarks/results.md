# Benchmark results

Numbers produced with the built-in suite (`van bench`, see [benchmarks/README.md](../../benchmarks/README.md)).
Voice-to-voice latency is measured **on the call recording** (end of the caller's speech →
first 10 ms frame that starts ≥ 100 ms of agent speech), not by summing component timings.
Every run writes a manifest (git SHA, lockfile hash, package versions, CPU/GPU, OS, scenario hash).

## T1 · Fully local cascade on a desktop CPU (2026-09-24)

* **Hardware:** AMD Ryzen 5 5600 (6 cores / 12 threads, `powersave` governor), 32 GB RAM, Linux 7.2 (CachyOS). The RTX 5070 Ti was **not** used (CPU-only run).
* **Stack:** Silero VAD v6.2 · Smart Turn v3.2 (int8) · faster-whisper `base` (CPU int8, English) · Ollama `LiquidAI/lfm2.5-1.2b-instruct` (temperature 0) · Kokoro v1.0 fp16 (CPU).
* **Scenario:** `benchmarks/scenarios/latency-local.yaml` — six customer-service questions voiced by Kokoro (`am_adam`), real-time paced; agent asked to answer in two or three full sentences; first turn reported separately as cold start; 5 measured turns.

| configuration | v2v p50 | v2v p90 | TTS first audio p50 | end-of-turn delay p50 | LLM TTFT p50 | dead air (> 2 s) |
|---|---:|---:|---:|---:|---:|---:|
| whole first sentence to TTS | 2,028 ms | 2,680 ms | 1,244 ms | ~620 ms | ~12 ms | 60 % |
| **first clause to TTS** (default since `efde03d`) | **1,128 ms** | **1,625 ms** | **408 ms** | ~620 ms | ~12 ms | **0 %** |

Where the time goes (default configuration): end-of-turn ≈ 620 ms (dominated by faster-whisper's
~360 ms CPU transcription of the final segment; Smart Turn runs concurrently), TTS first audio
≈ 400 ms (Kokoro renders one clause at RTF ≈ 0.16 on this CPU), LLM ≈ 12 ms, transport/playout
residual ≈ 100 ms.

### Improvements these runs led to

| change | effect |
|---|---|
| Trim TTS silence padding per sentence (Kokoro adds 40–120 ms before and ~530 ms after every sentence) | recording-vs-session residual 158 → 103 ms; no half-second gaps between sentences |
| Speak the first clause of a long first sentence on its own + flush the sentence adapter per segment | v2v p50 2,028 → 1,128 ms, dead air 60 % → 0 % |
| Audio turn detector runs concurrently with the STT flush | end-of-turn delay no longer = STT latency + detector latency |
| Pre-warming (`engine.warmup()`) | local LLM TTFT 582 ms cold → 10 ms warm |

### Next levers (tracked in #62 and the roadmap)

* GPU inference for STT and TTS (the RTX 5070 Ti was idle: faster-whisper float16 RTF ≈ 0.005, needs the CUDA 12 cuBLAS libraries).
* Streaming STT (sherpa-onnx / NeMo cache-aware models, #9) to cut the ~360 ms final-transcript wait.
* Speculative LLM generation on eager end-of-turn (#27).

## T7 · Framework overhead (mock components)

`van bench latency --engine mock` (energy VAD 0.4 s silence, no model latency): v2v p50 ≈ 402 ms
= 400 ms endpointing + ~2 ms; with `response_delay 0.3`: ≈ 702 ms; cascade of mocks ≈ 602 ms
(0.6 s VAD-only endpointing). Recording-vs-session residual ≈ 0.1 ms — the runtime adds
essentially nothing on top of its components.
