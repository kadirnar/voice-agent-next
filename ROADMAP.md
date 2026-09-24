# Roadmap

voice-agent-next is built in milestones. Every item is a GitHub issue; each issue is implemented on its own branch (`feat/<issue>-<slug>`) and merged through a pull request with green CI. Design rationale lives in [docs/research/REPORT.md](docs/research/REPORT.md); the architecture in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

**Vision:** the most complete open-source toolkit for real-time speech-to-speech agents — every architecture (native S2S, full-duplex, cascade, half-cascade) behind one engine interface; local and cloud providers; the same experience on Linux, macOS and Windows; and a benchmark suite that compares them on identical stimuli.

## M0 · Foundation — ✅ done

- [x] Audio primitives: `AudioFrame`, buffers, chunking, streaming resampler (soxr / numpy), bit-exact G.711, WAV I/O
- [x] Component interfaces + streaming base classes: STT (+ `StreamAdapter`), TTS (+ `SentenceStreamAdapter`), LLM, VAD state machine, turn detector
- [x] `S2SEngine` / `EngineConnection` + event protocol; `CascadeEngine` (VAD → STT → turn detector → LLM → TTS, half-cascade)
- [x] `AgentSession`: real-time playout pacing, barge-in + truncation to what was heard, tool rounds, history, per-turn voice-to-voice metrics
- [x] Provider registry (`provider/model` specs, lazy imports, entry points), config files, function tools, mock providers, energy VAD, loopback & file transports, model download cache
- [x] `van` CLI (`providers`, `doctor`, `devices`, `run`, `demo`), CI (Linux on PRs; macOS + Windows on `main`)
- [x] Research: six notes + synthesized report


## M1 · Local voice agent on every OS

Fully offline agent with local audio, VAD/turn detection, STT, LLM and TTS on all three OSes.

- [ ] #1 Local audio transport (microphone + speakers) with device selection `P0` — 🚧 wave 1
- [ ] #2 Echo cancellation & noise suppression processors (WebRTC APM) + half-duplex fallback `P0` — 🚧 wave 1
- [ ] #3 Silero VAD v6 provider (ONNX, no torch) `P0` — 🚧 wave 1
- [ ] #4 Smart Turn v3.2 end-of-turn detector (ONNX, audio-based) `P0` — 🚧 wave 1
- [ ] #5 faster-whisper STT provider (local CPU/CUDA) `P0` — 🚧 wave 1
- [ ] #6 Kokoro TTS provider (kokoro-onnx, local) `P0` — 🚧 wave 1
- [ ] #7 OpenAI & OpenAI-compatible LLM provider (Ollama, llama.cpp, vLLM, LM Studio, Groq, Cerebras, Together, OpenRouter…) `P0` — 🚧 wave 1
- [ ] #8 Presets and `van run` UX (local-cpu, local-gpu, apple, cloud-fast, openai-realtime, gemini-live) `P1`
- [ ] #9 sherpa-onnx providers: streaming STT (Zipformer / Parakeet / Moonshine), TTS and VAD `P1`
- [ ] #10 Apple Silicon providers via MLX (parakeet-mlx / mlx-whisper STT, mlx-audio TTS, mlx-lm) `P2`

## M2 · Native speech-to-speech engines

Native speech-to-speech engines, cloud and local.

- [ ] #11 OpenAI Realtime engine (WebSocket) with compatibility profiles (Azure, xAI, Qwen-Omni, vLLM, Speaches…) `P0` — 🚧 wave 1
- [ ] #12 Gemini Live engine (BidiGenerateContent) with session resumption and non-blocking tools `P0` — 🚧 wave 1
- [ ] #13 Moshi / PersonaPlex full-duplex engine (local moshi-server protocol) `P1`
- [ ] #14 Audio-input LLMs for half-cascades (gpt-audio, Qwen-Omni, vLLM-Omni, llama.cpp audio) `P1`
- [ ] #15 OpenAI GPT-Live engine (Live protocol, full-duplex, delegation) `P1`
- [ ] #16 Amazon Nova 2 Sonic engine (Bedrock bidirectional stream, 8-min rotation) `P2`
- [ ] #17 Engine session rotation & reconnect with context carry-over `P1`

## M3 · Cloud cascade providers

Cloud providers for cascades.

- [ ] #18 Deepgram: Nova-3 & Flux streaming STT (turn events) + Aura-2 TTS `P0` — 🚧 wave 1
- [ ] #19 Cartesia: Sonic TTS (WebSocket continuations, word timestamps) + Ink STT `P0` — 🚧 wave 1
- [ ] #20 AssemblyAI Universal-Streaming v3 STT (neural turn detection, ForceEndpoint) `P1`
- [ ] #21 ElevenLabs: Flash v2.5 / v3 TTS (stream-input, alignment) + Scribe v2 Realtime STT `P1`
- [ ] #22 Anthropic Claude LLM provider (streaming tool use, prompt caching) `P0` — 🚧 wave 1
- [ ] #23 Google Gemini LLM + Gemini TTS providers (google-genai) `P1`
- [ ] #24 OpenAI STT (realtime transcription) and TTS (gpt-4o-mini-tts) providers `P1`
- [ ] #25 Soniox and Speechmatics streaming STT providers `P2`

## M4 · Turn-taking & session intelligence

What makes conversations feel natural and robust.

- [ ] #26 Interruption policy: min duration/words, backchannel filter, false-interruption pause & resume `P0` — 🚧 wave 1
- [ ] #27 Speculative (preemptive) LLM generation in the cascade on eager end-of-turn `P1`
- [ ] #28 Tool-call watchdog fillers, non-blocking tools and delegation `P1`
- [ ] #29 Session recording (stereo WAV + JSONL timeline) and OpenTelemetry tracing `P1`
- [ ] #30 Provider failover chains (FallbackSTT / FallbackLLM / FallbackTTS) `P1`
- [ ] #31 Dynamic endpointing and dictation mode `P2`
- [ ] #32 Multi-agent handoffs and conversation flows `P2`

## M5 · Transports & serving

Getting audio in and out: browsers, phones, servers.

- [ ] #33 WebSocket server transport + browser client demo `P0` — 🚧 wave 1
- [ ] #34 WebRTC transport (aiortc peer-to-peer) + browser demo `P1`
- [ ] #35 Telephony serializers: Twilio Media Streams, Telnyx, Vonage, Plivo `P1`
- [ ] #36 OpenAI-Realtime-compatible server: serve any engine at /v1/realtime `P1`
- [ ] #37 `van serve`: worker pool with prewarm, health checks and concurrency limits `P2`
- [ ] #38 Docker images (CPU / CUDA) and compose files for fully local stacks `P2`

## M6 · Benchmark suite

Seven tracks x three tiers, measured at the audio boundary.

- [ ] #39 Benchmark harness core + T1 latency track (`van bench`) `P0` — 🚧 wave 1
- [ ] #40 T7 framework-overhead track + CI benchmark regression gate `P1`
- [ ] #41 T2 ASR track: WER/CER, RTFx, TTFS (LibriSpeech / FLEURS smoke subsets) `P1`
- [ ] #42 T3 TTS track: TTFA, RTF, round-trip WER, MOS predictors `P1`
- [ ] #43 T4 VAD & turn-taking track (eot-bench adapter, VAD frame metrics, barge-in battery) `P1`
- [ ] #44 T5 speech-to-speech quality track (Big Bench Audio, VoiceBench subsets) `P2`
- [ ] #45 T6 tool-use track (scripted scenarios with deterministic mock tools; τ-Voice adapter) `P2`

## M7 · Developer experience & release

Docs, examples, tooling, releases.

- [ ] #46 Documentation site (mkdocs-material) with guides and API reference `P1`
- [ ] #47 Example gallery (offline local agent, OpenAI Realtime, Gemini Live, telephony, tools, benchmarks) `P1`
- [ ] #48 Model manager: `van models` (list/download/verify/prune cached models) `P1`
- [ ] #49 Hardware-aware backend auto-selection (CUDA / TensorRT / CoreML / DirectML / MLX / CPU) `P2`
- [ ] #50 `van doctor` deep diagnostics (PortAudio host APIs, echo test, mic level meter, latency probe) `P2`
- [ ] #51 Release automation: PyPI trusted publishing, changelog, versioning `P2`

## How work is scheduled

Work proceeds in **waves**: every wave takes the highest-priority issues whose dependencies are merged, and runs them in parallel (one agent per issue, each in its own git worktree and branch). After each wave the maintainer reviews and merges the PRs, re-runs the full test suite and the benchmark smoke tier, analyses gaps, and files follow-up issues for the next wave.

- **Wave 1 (in progress):** local audio + echo cancellation, Silero VAD, Smart Turn, faster-whisper, Kokoro, OpenAI-compatible LLMs, OpenAI Realtime, Gemini Live, Deepgram, Cartesia, Anthropic, interruption policy, WebSocket transport, benchmark harness + latency track.
- **Next:** presets, sherpa-onnx, remaining cloud providers, speculative generation, async tools, recording/tracing, failover, WebRTC, telephony, remaining benchmark tracks, docs site.
