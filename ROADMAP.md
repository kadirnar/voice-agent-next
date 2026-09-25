# Roadmap

voice-agent-next is built in milestones. Every item is a GitHub issue; each issue is implemented on its own branch (`feat/<issue>-<slug>`) and merged through a pull request with green CI. Design rationale lives in [docs/research/REPORT.md](docs/research/REPORT.md); the architecture in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

**Vision:** the most complete open-source toolkit for real-time speech-to-speech agents — every architecture (native S2S, full-duplex, cascade, half-cascade) behind one engine interface; local and cloud providers; the same experience on Linux, macOS and Windows; and a benchmark suite that compares them on identical stimuli.

## M0 · Foundation — ✅ done

- [x] Audio primitives: `AudioFrame`, buffers, chunking, streaming resampler (soxr / numpy), bit-exact G.711, WAV I/O
- [x] Component interfaces + streaming base classes: STT (+ `StreamAdapter`), TTS (+ `SentenceStreamAdapter`), LLM, VAD state machine, turn detector
- [x] `S2SEngine` / `EngineConnection` + event protocol; `CascadeEngine` (VAD → STT → turn detector → LLM → TTS, half-cascade)
- [x] `AgentSession`: real-time playout pacing, barge-in + truncation to what was heard, tool rounds, history, per-turn voice-to-voice metrics
- [x] Provider registry (`provider/model` specs, lazy imports, entry points), config files, function tools, mock providers, energy VAD, loopback & file transports, model download cache
- [x] `van` CLI (`providers`, `doctor`, `devices`, `run`, `demo`), CI (Linux + Windows on every PR; macOS on `main` and `ci:full`)
- [x] Research: six notes + synthesized report


## M1 · Local voice agent on every OS

Fully offline agent with local audio, VAD/turn detection, STT, LLM and TTS on all three OSes. A fully local pipeline (Silero + Smart Turn + faster-whisper + Ollama + Kokoro) runs end to end today: 1.1 s voice-to-voice p50 on a desktop CPU ([results](docs/benchmarks/results.md)).

- [x] #1 Local audio transport (microphone + speakers) with device selection `P0`
- [x] #2 Echo cancellation & noise suppression processors (WebRTC APM) + half-duplex fallback `P0`
- [x] #3 Silero VAD v6 provider (ONNX, no torch) `P0`
- [x] #4 Smart Turn v3.2 end-of-turn detector (ONNX, audio-based) `P0`
- [x] #5 faster-whisper STT provider (local CPU/CUDA) `P0`
- [x] #6 Kokoro TTS provider (kokoro-onnx, local) `P0`
- [x] #7 OpenAI & OpenAI-compatible LLM provider (Ollama, llama.cpp, vLLM, LM Studio, Groq, Cerebras, Together, OpenRouter…) `P0`
- [x] #8 Presets and `van run` UX (local-cpu, local-gpu, apple, cloud-fast, openai-realtime, gemini-live) `P1`
- [x] #9 sherpa-onnx providers: streaming STT (Zipformer / Parakeet / Moonshine), TTS and VAD `P1`
- [x] #10 Apple Silicon providers via MLX (parakeet-mlx / mlx-whisper STT, mlx-audio TTS, mlx-lm) `P2`
- [x] #76 Kyutai Pocket TTS: audio-streaming local TTS on CPU with voice cloning `P1`
- [x] #77 Moonshine Streaming STT (moonshine-voice): low-latency local STT for CPUs and edge devices `P1`
- [x] #78 NeMo-Speech.cpp server: Nemotron cache-aware streaming STT, Parakeet, Magpie TTS `P1`
- [x] #79 Local GPU TTS: Chatterbox (Turbo/Nano/Multilingual), Qwen3-TTS, CosyVoice 3 `P2`

## M2 · Native speech-to-speech engines

Native speech-to-speech engines, cloud and local.

- [x] #11 OpenAI Realtime engine (WebSocket) with compatibility profiles (Azure, xAI, Qwen-Omni, vLLM, Speaches…) `P0`
- [x] #12 Gemini Live engine (BidiGenerateContent) with session resumption and non-blocking tools `P0`
- [x] #13 Moshi / PersonaPlex full-duplex engine (local moshi-server protocol) `P1`
- [ ] #14 Audio-input LLMs for half-cascades (gpt-audio, Qwen-Omni, vLLM-Omni, llama.cpp audio) `P1`
- [x] #15 OpenAI GPT-Live engine (Live protocol, full-duplex, delegation) `P1`
- [x] #16 Amazon Nova 2 Sonic engine (Bedrock bidirectional stream, 8-min rotation) `P2`
- [x] #17 Engine session rotation & reconnect with context carry-over `P1`
- [x] #75 Omni models in the cascade: audio-output LLMs (LFM2.5-Audio via llama-liquid-audio-server, gpt-audio, Qwen-Omni) — fully local native S2S on CPU `P1`

## M3 · Cloud cascade providers

Cloud providers for cascades.

- [x] #18 Deepgram: Nova-3 & Flux streaming STT (turn events) + Aura-2 TTS `P0`
- [x] #19 Cartesia: Sonic TTS (WebSocket continuations, word timestamps) + Ink STT `P0`
- [x] #20 AssemblyAI Universal-Streaming v3 STT (neural turn detection, ForceEndpoint) `P1`
- [x] #21 ElevenLabs: Flash v2.5 / v3 TTS (stream-input, alignment) + Scribe v2 Realtime STT `P1`
- [x] #22 Anthropic Claude LLM provider (streaming tool use, prompt caching) `P0`
- [x] #23 Google Gemini LLM + Gemini TTS providers (google-genai) `P1`
- [x] #24 OpenAI STT (realtime transcription) and TTS (gpt-4o-mini-tts) providers `P1`
- [x] #25 Soniox and Speechmatics streaming STT providers `P2`

## M4 · Turn-taking & session intelligence

What makes conversations feel natural and robust.

- [x] #26 Interruption policy: min duration/words, backchannel filter, false-interruption pause & resume `P0`
- [x] #27 Speculative (preemptive) LLM generation in the cascade on eager end-of-turn `P1`
- [x] #28 Tool-call watchdog fillers, non-blocking tools and delegation `P1`
- [x] #29 Session recording (stereo WAV + JSONL timeline) and OpenTelemetry tracing `P1`
- [x] #30 Provider failover chains (FallbackSTT / FallbackLLM / FallbackTTS) `P1`
- [x] #31 Dynamic endpointing and dictation mode `P2`
- [x] #32 Multi-agent handoffs and conversation flows `P2`

- [x] #105 Spoken-form text normalization for TTS input (numbers, currency, dates, URLs) `P1`
- [x] #113 Turn-taking quality on the local cascade (premature replies, backchannel barge-ins, dead air) `P1`

- [x] #124 Semantic (text) end-of-turn detector fused with the audio detector `P1`

## M5 · Transports & serving

Getting audio in and out: browsers, phones, servers.

- [x] #33 WebSocket server transport + browser client demo `P0`
- [x] #34 WebRTC transport (aiortc peer-to-peer) + browser demo `P1`
- [x] #35 Telephony serializers: Twilio Media Streams, Telnyx, Vonage, Plivo `P1`
- [x] #36 OpenAI-Realtime-compatible server: serve any engine at /v1/realtime `P1`
- [x] #37 `van serve`: worker pool with prewarm, health checks and concurrency limits `P2`
- [x] #38 Docker images (CPU / CUDA) and compose files for fully local stacks `P2`

## M6 · Benchmark suite

Seven tracks x three tiers, measured at the audio boundary.

- [x] #39 Benchmark harness core + T1 latency track (`van bench`) `P0`
- [x] #40 T7 framework-overhead track + CI benchmark regression gate `P1`
- [x] #41 T2 ASR track: WER/CER, RTFx, TTFS (LibriSpeech / FLEURS smoke subsets) `P1`
- [x] #42 T3 TTS track: TTFA, RTF, round-trip WER, MOS predictors `P1`
- [x] #43 T4 VAD & turn-taking track (eot-bench adapter, VAD frame metrics, barge-in battery) `P1`
- [x] #44 T5 speech-to-speech quality track (Big Bench Audio, VoiceBench subsets) `P2`
- [x] #45 T6 tool-use track (scripted scenarios with deterministic mock tools; τ-Voice adapter) `P2`

## M7 · Developer experience & release

Docs, examples, tooling, releases.

- [x] #46 Documentation site (mkdocs-material) with guides and API reference `P1`
- [x] #47 Example gallery (offline local agent, OpenAI Realtime, Gemini Live, telephony, tools, benchmarks) `P1`
- [x] #48 Model manager: `van models` (list/download/verify/prune cached models) `P1`
- [x] #49 Hardware-aware backend auto-selection (CUDA / TensorRT / CoreML / DirectML / MLX / CPU) `P2`
- [x] #50 `van doctor` deep diagnostics (PortAudio host APIs, echo test, mic level meter, latency probe) `P2`
- [x] #51 Release automation: PyPI trusted publishing, changelog, versioning `P2`

## How work is scheduled

Work proceeds in **waves**: every wave takes the highest-priority issues whose dependencies are merged, and runs them in parallel (one agent per issue, each in its own git worktree and branch). After each wave the maintainer reviews and merges the PRs, re-runs the full test suite and the benchmark smoke tier, analyses gaps, and files follow-up issues for the next wave.

- **Wave 1 (done, PRs #53–#68):** local audio + echo cancellation, Silero VAD, Smart Turn, faster-whisper, Kokoro, OpenAI-compatible LLMs (11 servers/clouds), OpenAI Realtime (+ Azure, xAI, Qwen-Omni, vLLM, Speaches, LocalAI profiles), Gemini Live, Deepgram, Cartesia, Anthropic, interruption policy, WebSocket transport, benchmark harness + latency track. Integration follow-ups: #62 (session pre-warm and history ordering landed in #69).
- **Wave 2 (done):** sherpa-onnx streaming STT/TTS (#9), AssemblyAI (#20), ElevenLabs (#21), Gemini LLM/TTS (#23), OpenAI STT/TTS (#24), speculative generation (#27), recording + tracing (#29), failover (#30), telephony (#35), OpenAI-Realtime-compatible server (#36), overhead track + CI gate (#40), GPU/hardware backends (#49).
- **Wave 3 (done):** presets (#8), async tools (#28), WebRTC (#34), model manager (#48), Pocket TTS (#76), Moonshine (#77), omni models / LFM2.5-Audio (#75), Moshi (#13), session rotation (#17), dynamic endpointing (#31), handoffs (#32), `van serve` pool (#37), Docker (#38), ASR/TTS/turn-taking tracks (#41–#43), docs site (#46), examples (#47), `van doctor` (#50), GPU TTS (#79), Soniox/Speechmatics (#25), text normalization (#105); CI stabilization on Windows/macOS (#114).
- **Wave 4 (done):** turn-taking quality (#113) and semantic end-of-turn fusion (#124), MLX on Apple Silicon (#10), GPT-Live (#15), Nova 2 Sonic (#16), NeMo-Speech.cpp (#78), T5 quality + T6 tool-use tracks (#44, #45), release automation (#51).
- **Next:** CI flake hunt, the preset default for fused turn detection, a larger local LLM default (T5/T6 show the 1.2B model is the quality bottleneck), and the first public release when the owner decides.
- **Lessons that shaped the process:** agents commit and push work in progress (an API limit once killed half a wave; the recovered agents resumed from their pushed branches), Windows CI runs on every PR (two merged PRs had Windows-only test races), and the benchmark drives the backlog (the first local run found a 0.9 s win in how the first sentence is split for TTS).
