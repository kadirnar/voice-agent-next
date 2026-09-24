# Voice Agents in 2026 — Research Report

*voice-agent-next research synthesis · 2026-09-24*

This report condenses six detailed research notes (~40,000 words, ~650 cited primary sources: docs, model cards, papers, repositories, leaderboards) into the findings that shape **voice-agent-next**. Every claim below is traceable to a note; numbers are vendor-reported unless the note says otherwise.

| # | Note | Scope |
|---|---|---|
| 01 | [Architectures & frameworks](01-architectures-and-frameworks.md) | cascade / half-cascade / realtime / full-duplex; Pipecat, LiveKit Agents, HF speech-to-speech, Unmute, TEN, FastRTC, OpenAI Agents SDK, hosted platforms |
| 02 | [Speech-to-speech models](02-speech-to-speech-models.md) | OpenAI Realtime & GPT-Live, Gemini Live, Nova Sonic, Grok, Qwen-Omni; Moshi, PersonaPlex, MiniCPM-o, Qwen3-Omni… |
| 03 | [STT / TTS / LLM landscape](03-stt-tts-llm-landscape.md) | cloud & local recognizers, synthesizers and LLMs for voice; default stacks |
| 04 | [Turn-taking, VAD, interruptions](04-turn-taking-vad-interruptions.md) | VAD, end-of-turn models, endpointing, barge-in, echo cancellation, full-duplex research |
| 05 | [Latency, transports, production](05-latency-transports-production.md) | latency budgets & measurement, WebRTC/WebSocket/telephony, local audio I/O per OS, deployment, reliability |
| 06 | [Evaluation & benchmarks](06-evaluation-and-benchmarks.md) | VoiceBench, Big Bench Audio, Full-Duplex-Bench, τ-Voice, EVA, eot-bench…; proposed benchmark suite |

---

## Executive summary

1. **There is no single "best architecture".** Tuned streaming **cascades** (VAD → STT → LLM → TTS) still lead on answer quality, tool use, controllability and cost, and reach ~400–800 ms voice-to-voice; **native speech-to-speech** models lead on turn-taking naturalness and prosody; **full-duplex + delegation** ("talker/thinker", e.g. OpenAI GPT-Live-1, Sep 2026) is the new frontier. A library must treat all of them as interchangeable *engines* with declared capabilities — and benchmark them against each other. *(01, 02, 06)*
2. **The OpenAI Realtime event protocol is the de facto wire standard.** One well-built client with "compat profiles" covers ~10 backends (OpenAI, Azure OpenAI, Azure Voice Live, xAI Grok, Qwen-Omni-Realtime, vLLM, Unmute, Speaches, LocalAI, LiteLLM); Gemini Live, Nova Sonic, GPT-Live and Moshi need dedicated adapters. *(02)*
3. **Turn-taking is now a layered ML subsystem**, not a silence timer: VAD → semantic end-of-turn model → (optional LLM markers) → interruption classifier → speculative generation → false-interruption recovery. Silence-only endpointing cuts users off 55.6% of the time at a 300 ms budget vs 9.9–35% for learned detectors (eot-bench, vendor-run). *(04)*
4. **Licenses, not accuracy, decide the default turn-taking stack.** LiveKit's turn detectors may only be used with LiveKit Agents; TEN's VAD/turn models carry non-compete clauses; Vogent's weights can't be a platform default. **Silero VAD (MIT) + Smart Turn v3.2 (BSD-2, 8 MB, 23 languages)** is the only fully permissive local stack. *(01, 04)*
5. **Cloud STT absorbed turn detection.** Deepgram Flux, AssemblyAI U3.5, Cartesia Ink-2, Speechmatics Linden, xAI and Meta Muse emit start/eager-end/resumed/end-of-turn events — the STT interface must model them. *(03)*
6. **Latency must be measured acoustically.** Typical cloud cascades land at 1.1–1.3 s voice-to-voice; the best local stack reported is 508 ms p50 (RTX 5090). Component sums mis-predict real silence by 0.27–0.7 s; ground truth is a recording measured from end of user speech to agent speech onset. *(05, 06)*
7. **Local, cross-platform use is the weakest area of existing frameworks.** Of the major frameworks only LiveKit tests on Windows and only it ships echo cancellation for local mic+speaker; HF s2s tells users to wear headphones; Unmute is Linux/WSL-only; TEN is Python 3.10-only. *(01, 05)*
8. **Echo cancellation is the key technical risk for local audio.** The practical cross-platform route is the WebRTC Audio Processing Module shipped in the `livekit` wheel (Apache-2.0, wheels for Linux/macOS/Windows); OS-level AEC is inconsistent. *(04, 05)*
9. **Agentic voice is the hardest capability.** On τ-Voice voice agents solve 26–51% of tasks vs ~85% for text agents; EVA-Bench finds no system above 0.5 on both accuracy and experience. Tool use deserves first-class runtime support (async tools, fillers, delegation) and its own benchmark track. *(06)*
10. **Open full-duplex models exist but are not yet safe defaults** (Moshi takes the floor in 98.5% of user pauses; Moshi and Freeze-Omni rank near the bottom on interruption handling). Keep them available and wrap them in the same guards. *(02, 04)*

---

## 1. Architectures

| | Cascaded | Half-cascade | Realtime S2S (turn-based) | Full-duplex S2S (+ delegation) |
|---|---|---|---|---|
| Path | audio → VAD/turn → STT → LLM → TTS | audio → audio-LLM → text → TTS | audio ↔ one model, server VAD | continuous audio in/out, model owns the floor, reasoning/tools delegated |
| Examples | Pipecat/LiveKit pipelines, HF s2s, Unmute | Ultravox, Gemma 4 / Voxtral audio-in + TTS | gpt-realtime-2.1, Gemini 3.8 Live, Nova 2 Sonic, Grok Voice | GPT-Live-1, Moshi, PersonaPlex, MiniCPM-o 4.5 |
| Latency | ~400–800 ms tuned | moderate | "fastest" in theory; slower than a tuned cascade in practice (primer) | Moshi ~200 ms; GPT-Live 0.8 s turn-taking (reported) |
| Scripted speech (`say`) | yes | yes | no | no |
| Tools | mature | less mature | less mature | via delegation |
| Transcripts | full | output only | delayed user transcripts | after the audio |
| Cost | lowest | mid | 3–5× a cascade (primer) | $0.05/min + backend |

**Trends (2025–2026):** frameworks now name all four patterns explicitly (LiveKit, OpenAI's voice-agent guide); the talker/thinker split is converging across stacks (GPT-Live delegation, Realtime-Venus `<delegate>`, MoshiRAG, Pipecat async tools, LiveKit subagents); turn-taking is layered; testing/evals move into frameworks (Pipecat Evals, LiveKit test framework).

**Framework landscape:** Pipecat (BSD-2, frames + processors, broadest provider catalog, Ubuntu-only CI, no local AEC) and LiveKit Agents (Apache-2.0, `AgentSession` + nodes, cross-platform CI and local AEC, but best turn/interruption models locked to its cloud/license) dominate; HF speech-to-speech 1.0 is a cascade server speaking the OpenAI Realtime protocol; Unmute (Kyutai) is Linux/WSL + CUDA; TEN's license forbids on-device hosting; Vocode and FastRTC have stalled. Recurring pain points: silent failures, echo/self-interruption, cloud-locked features, API churn, too many knobs without presets, platform friction. *(01 §3–5)*

## 2. Native speech-to-speech models and realtime APIs

**Cloud (Sep 2026):**

| API | Protocol | Audio in → out | Turn detection | Price | Limits |
|---|---|---|---|---|---|
| OpenAI Realtime (gpt-realtime-2.1 / -mini) | Realtime GA; WS, WebRTC, SIP | PCM16 24 kHz | server_vad, semantic_vad, manual | ≈$0.019 in / $0.077 out per min | 60 min, 128k ctx |
| OpenAI GPT-Live-1 | new `/v1/live/sessions` protocol | — | model-owned full duplex | $0.05/min + backend | 128k |
| Gemini 3.8 Live | BidiGenerateContent WS | PCM16 **16 kHz** → 24 kHz | automatic activity detection | $0.005 in / $0.018 out per min | ~10 min connections, resumption 2 h |
| Amazon Nova 2 Sonic | Bedrock HTTP/2 bidi stream | 16 → 24 kHz | model + endpointing sensitivity | ~$3/$12 per 1M (unverified) | **8-min connection** |
| xAI Grok Voice | Realtime-compatible | 8–48 kHz, G.711, Opus | server_vad | $0.08/min | — |
| Qwen-Omni-Realtime | Realtime-style (beta names) | 16 → 24 kHz | server/semantic VAD | — | 120 min |

Independent snapshot (Artificial Analysis S2S index): Gemini 3.8 Live Extended Thinking 82.6, GPT-Live-1 81.5, Grok Voice Think Fast 2.0 81.3 (fastest TTFA 0.70 s), GPT-Realtime-2 73.6.

**Open weights:** Moshi (7B, CC-BY, full-duplex, 24 GB bf16 / MLX q4), NVIDIA PersonaPlex-7B (Moshi-based, commercial OK), MiniCPM-o 4.5 (9B, Apache-2.0, full-duplex, 11 GB int4 via llama.cpp), Realtime-Venus (delegation), Qwen3-Omni-30B-A3B (best open omni, needs ≥68 GB), LFM2.5-Audio-1.5B (CPU-capable, English), speech-in/text-out models (Ultravox, Voxtral, Gemma 4, Nemotron 3 Nano Omni).

**Engineering implications:** carry the sample rate with every frame and resample once per edge; treat session limits and rotation (GoAway, resumption, re-seeding from transcripts) as first-class; track *played* audio for truncation; make tool execution asynchronous with cancellation (Gemini `toolCallCancellation`; GPT-Live does not cancel backend work on barge-in). *(02 §6)*

## 3. Cascade components

**STT.** Most accurate streaming: xAI Grok Voice Transcribe 2.0 (2.73% AA-WER), Meta Muse, Cartesia Ink-2, ElevenLabs Scribe v2 RT; fastest finals: Deepgram Flux (21 ms), Soniox v5 (54 ms), Nova-3; cheapest: Soniox ($0.002/min). OpenAI realtime transcription is slow to finalize (~0.7–0.8 s). Local: NVIDIA Nemotron streaming (cache-aware, via NeMo-Speech.cpp on all OSes), Parakeet TDT v3 (36× real-time on a desktop CPU), Qwen3-ASR-1.7B (most accurate open), Moonshine (edge), faster-whisper/whisper.cpp; `sherpa-onnx` runs everywhere. Forced finalization ("flush") is how external endpointing reaches finals in tens of ms.

**TTS.** Cloud: Cartesia Sonic 3.6 (#1 AA arena, word timestamps, WebSocket continuations), Gemini 3.8 Flash TTS, Inworld TTS-2 (Realtime-protocol WebSocket), ElevenLabs Flash v2.5 (~75 ms model latency), Deepgram Aura-2/Flux TTS; LMNT shut down. Local: Kokoro-82M (Apache-2.0, best small open model, no text streaming), Kyutai Pocket TTS (streaming + cloning on 2 CPU cores), Chatterbox Turbo/Flash (MIT), Qwen3-TTS & CosyVoice 3 (Apache-2.0); most top-rated open weights are non-commercial (Fish S2, Voxtral TTS, Breeze, Higgs) or GPL (Piper).

**LLMs for voice** (Pipecat voice-readiness, 30-turn tool-use, ~700 ms budget): Qwen3.8-27B 98.2% @ 649 ms hosted and **97.8% @ 101 ms on a local RTX 5090**; Claude Haiku 4.5 98.0% @ 637 ms; GPT-4.1 96.3%; Groq gpt-oss-120b 98 ms but 86.3%. Every local server (Ollama, llama.cpp, vLLM, LM Studio, mlx-lm) and fast host is OpenAI-compatible → one OpenAI-compatible client + native Anthropic and Gemini adapters cover the field.

**Recommended default stacks** *(03 §9.2)*

| Profile | STT | LLM | TTS |
|---|---|---|---|
| Local CPU | Moonshine streaming / Parakeet v3 (sherpa-onnx) | Qwen3.5-4B / Gemma 4 E4B (llama.cpp / Ollama) | Kokoro (kokoro-onnx), Pocket TTS |
| Local GPU 16 GB | Nemotron streaming (NeMo-Speech.cpp) / faster-whisper turbo | Qwen3.5-9B / Gemma 4 12B / gpt-oss-20b | Chatterbox-Turbo, Qwen3-TTS, CosyVoice 3 |
| Apple Silicon | parakeet-mlx / mlx-audio / whisper.cpp | mlx_lm / LM Studio / Ollama | Kokoro / Qwen3-TTS via mlx-audio |
| Cloud low-latency | Deepgram Flux / AssemblyAI U3.5 / Soniox | Claude Haiku 4.5, GPT-4.1, Groq/Cerebras | Cartesia Sonic 3.6, ElevenLabs Flash |

## 4. Turn-taking, VAD and interruptions

* **VAD:** Silero v6.2 (MIT, 32 ms windows at 16 kHz, <1 ms/chunk) — ship the ONNX file, since `pip install silero-vad` pulls torch; run ONNX Runtime single-threaded without spin-waiting.
* **End-of-turn:** Smart Turn v3.2 (audio, 8 MB int8, ~12 ms on CPU) is the permissive default; text/STT-built-in/realtime server detectors plug in behind the same interface. Smart Turn needs a *short* VAD stop time (0.2–0.25 s) — long stop times add their full length to every reply.
* **Endpointing policy matters as much as the model:** threshold + min delay (~0.3–0.4 s) + max hold (~2.5 s); dictation mode (longer); dynamic endpointing; speculative LLM start on eager end-of-turn (+50–70% LLM calls, hundreds of ms saved) whose output must never reach the speaker before confirmation.
* **Barge-in needs more than VAD:** VAD-only barge-in is 66% false positives (Krisp). Recommended: minimum duration/words, backchannel filtering, and **pause-then-resume** on possible false interruptions (resume if no words within 2 s).
* **Truncate to what was heard:** commit only the played portion of the agent's turn (OpenAI `conversation.item.truncate` with `audio_end_ms`; word timestamps where the TTS provides them); account for device output latency.
* **Echo:** WebRTC APM (livekit wheel) with the playback stream as reference and a delay hint; headphone mode and half-duplex (mic gated while speaking) as fallbacks.

## 5. Latency, transports and production

* **Budget:** reference WebRTC cascade = 1,293 ms (mic 40 · inbound transport 74 · STT+endpointing 300 · LLM TTFB 650 · aggregation 20 · TTS 120 · outbound 89); STT/endpointing + LLM ≈ 73%. PSTN adds ~230 ms (Twilio 1,115 ms mouth-to-ear target).
* **Levers:** stream everything; semantic turn detection; speculative generation; send the first clause to TTS immediately (Cartesia buffers up to 3 s by default); warm connections, same-region placement, prompt caching; tool-call watchdog fillers; avoid transcoding.
* **Transports:** WebRTC for internet clients (no head-of-line blocking, browser AEC); WebSocket for server-to-server and telephony media streams; SIP via providers. Telephony dialects differ (Twilio μ-law 8 kHz only; Telnyx/Vonage/Plivo 16 kHz L16) — normalize behind serializers with clear/mark/DTMF.
* **Local audio:** `sounddevice` (bundles PortAudio on Windows/macOS); Debian/Ubuntu's PortAudio lacks a PulseAudio/PipeWire host API (use ALSA `default`/`pipewire`); Windows WASAPI shared mode ~10 ms periods; nothing in PortAudio/miniaudio cancels echo.
* **Reliability:** failover chains with health probes; treat silent reconnects as failures (one Pipecat incident: 66 s of dead air); commit only spoken text.
* **Observability:** OpenTelemetry conversation → turn → STT/LLM/TTS spans; per-turn metrics with exact definitions (`endpointing_delay`, `stt_final_latency`, `llm_ttft`, `tts_ttfb` per utterance, `server_v2v`, `client_v2v`, barge-in stop latency, false-interruption rate, speculation waste, cost).
* **Cost:** cascades ~$0.07–0.13/min all-in vs speech-to-speech ~$0.18–0.21/min (vendor fleet data); self-hosting pays off only at high utilization.

## 6. Evaluation and benchmarks

Five layers — component, pipeline latency, conversational dynamics, spoken content, task success (+ human preference). Key public benchmarks: VoiceBench, Big Bench Audio, URO-Bench, VocalBench (content); Full-Duplex-Bench v1–v3, TurnBench, ECHO (dynamics); τ-Voice, EVA-Bench, VoiceAgentBench (agentic); LiveKit eot-bench (end-of-turn); Open ASR Leaderboard conventions (ASR); Seed-TTS-eval / UTMOSv2 / DNSMOS (TTS). Reusable OSS: tau2-bench, ServiceNow EVA, eot-bench, pipecat stt-benchmark, open_asr_leaderboard.

**Proposed suite (adopted, see ROADMAP M6):** one stimulus-driven harness measuring at the audio boundary (stereo recording = ground truth; traces explain it); seven tracks — **T1 latency, T2 ASR, T3 TTS, T4 VAD/turn-taking, T5 S2S quality, T6 tool use, T7 framework overhead**; three tiers — smoke (CI, CPU, no keys, ≤10 min), nightly, full; JSONL + summary JSON + Markdown/HTML + stereo WAV artifacts; reproducibility manifest (versions, hardware, region/RTT, dataset hashes, judge ids); fair-comparison rules for local vs cloud and native vs cascade (same stimuli, own endpointing, Pareto quality-vs-latency view).

---

## 7. Decisions for voice-agent-next

| Decision | Why (note) |
|---|---|
| One `S2SEngine`/`EngineConnection` interface + one event protocol for native S2S, cascades, half-cascades (and later full-duplex), with **declared capabilities** (`native_audio`, `truncation`, `tool_mode`, `max_session_duration`…) | Architectures differ in ways that must be explicit; enables cross-architecture benchmarking (01 §6.1, 02 §6.2) |
| The **session** owns barge-in, playback pacing, truncation-to-heard, tool execution, history and turn metrics — identical for every engine | Pipecat/LiveKit lessons; context must contain only what was heard (01, 04 §5.3, 05 §7.3) |
| Internal audio = s16le `AudioFrame` carrying its own rate; resample once at edges (soxr or numpy fallback); bit-exact G.711 | Formats vary (16/24 kHz, μ-law, Opus) (02 §6.3, 05 §9.1) |
| Default local turn-taking = Silero ONNX + Smart Turn v3.2, energy VAD fallback; VAD candidate pause 0.25 s, prefix 0.5 s; endpointing 0.4/2.5 s (detector) or 0.6 s (VAD only), measured from speech end | Only permissive stack; research defaults (04 §8.2) |
| STT events include `END_OF_TURN`, `EAGER_END_OF_TURN`, `TURN_RESUMED`; `flush()` = force-finalize | STT-built-in turn detection (03 §9.4) |
| Provider registry with `provider/model` specs, lazy imports, optional-dependency extras, entry-point plugins | "many models" without dependency hell (01 §6.3, 03 §9.3) |
| First engine targets: OpenAI-Realtime client with compat profiles → Gemini Live → Moshi/PersonaPlex local → chat-audio (half-cascade) → Nova Sonic, GPT-Live | Coverage per line of code (02 §6.1) |
| Local transport with WebRTC APM echo cancellation + headphone and half-duplex modes; `van doctor` detects PortAudio/Linux pitfalls | Weakest area of existing frameworks (01 §6.4, 04 §6, 05 §9.5) |
| Serve any engine as an **OpenAI-Realtime-compatible server** | De facto wire standard; reuse existing clients/SDKs (01, 02) |
| Built-in benchmark suite (7 tracks × 3 tiers), acoustic ground truth, CI regression gate | Component sums lie; no framework benchmarks architectures end to end (05 §2.4, 06 §8) |
| Presets (`local-cpu`, `local-gpu`, `apple`, `cloud-fast`, `openai-realtime`, `gemini-live`…) instead of knob soup | Pain point in every framework (01 §5) |
| Model licenses tracked in the registry; non-commercial models behind an explicit opt-in | Licenses decide defaults (03 §9.4, 04 §8.6) |

## 8. Differentiators we are building toward

1. The same local experience on **Linux, macOS and Windows** (CI matrix, bundled echo cancellation, hardware-aware backends: CUDA/ONNX/CPU/MLX).
2. A **cross-architecture benchmark** harness: cascade vs half-cascade vs realtime vs full-duplex on identical stimuli, reporting v2v p50/p95/p99 at playback, false-cutoff vs endpointing delay, barge-in latency, context accuracy after interruptions, tool-call correctness and cost per minute.
3. A **fully open turn-taking and interruption stack** (no cloud or license lock-in).
4. **Delegation/async tools that work with any engine**, not just GPT-Live.
5. **Standards-based serving** (OpenAI Realtime compatibility).

## 9. Risks and open questions

* Echo cancellation quality varies per device/OS — must be tested on real hardware; half-duplex fallback required.
* Full-duplex models break turn-shaped abstractions (append-only context, model-owned barge-in) — the event model needs a duplex extension.
* Session limits (Gemini ~10 min connections, Nova 8 min) require transparent rotation with context carry-over.
* onnxruntime ≥ 1.24 dropped Intel-Mac wheels; `kokoro` requires Python < 3.13; vLLM is Linux-only — packaging must degrade gracefully per platform.
* Vendor benchmarks crown their own vendors — ship defaults backed by our own benchmark runs.
* Several evaluation datasets are non-commercial or unlicensed — download at run time, never redistribute.
