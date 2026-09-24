# 03 — Streaming STT, TTS and LLM landscape for cascaded voice agents

*Research snapshot for **voice-agent-next**. Compiled 2026-09-24. Scope: cloud and local speech-to-text (STT), text-to-speech (TTS) and LLMs for a cascaded, real-time voice agent (VAD/turn detection → STT → LLM → TTS).*

> **Method and caveats.** Sources are vendor docs and pricing pages, model cards, PyPI and GitHub metadata, and independent leaderboards (Artificial Analysis, the Hugging Face Open ASR Leaderboard, Pipecat's benchmarks, TTS Arena V2 and BFCL). Where a leaderboard page embeds its data as JSON or CSV, the numbers below were read from that data rather than from summaries. A latency figure marked **vendor** is the provider's own claim. An independent figure names its benchmark. Prices are list or pay-as-you-go in USD as shown on 2026-09-24. Several are promotional and all change often. Anything I could not confirm from a primary source is marked *(unverified)*.

---

## 1. Executive summary

- **Turn detection now lives inside STT.** Deepgram Flux, Cartesia Ink-2, AssemblyAI Universal-3.5 Pro Realtime, xAI Grok Voice Transcribe ("Smart Turn"), Meta Muse Voice Transcribe and Speechmatics Agent STT all emit start-of-turn and end-of-turn events, and some emit an *eager* end-of-turn [16][51][26][66][65][46]. The STT abstraction in voice-agent-next needs events for `turn_started`, `eager_end_of_turn`, `turn_resumed` and `end_of_turn`, not just `partial` and `final`.
- **Two independent streaming-STT benchmarks exist, and they measure different things.** Artificial Analysis (AA) scores accuracy on agent-style audio plus latency from the end of speech [1]. Pipecat scores *semantic* WER and TTFS through a real agent pipeline [9]. Leaders:
  - Most accurate: Grok Voice Transcribe 2.0 (2.73 % AA-WER), Muse (3.06 %), Ink-2 (3.36 %) and Scribe v2 Realtime (3.59 %) [1]. On Pipecat: Muse (0.83 % semantic WER), AssemblyAI U3.6/U3.5 Pro (0.96 % / 1.22 %) and Speechmatics Linden (1.05 %) [9].
  - Fastest: Deepgram Flux reaches a final transcript 0.021 s after speech ends and Soniox v5 in 0.054 s [1]. NVIDIA Nemotron 3 ASR has a 221 ms median TTFS [9].
  - Cheapest: Modulate ($0.0008/min, 4.77 % AA-WER) and Inworld STT 1 ($0.0014/min, 4.18 %) on AA [1], then Soniox ($0.002/min), AssemblyAI Universal-Streaming ($0.0025/min) and Muse ($0.003/min) [44][24][65].
- **OpenAI replaced its transcription lineup.** `gpt-live-transcribe` and `gpt-transcribe` arrived in July 2026 [39]. Removal of `whisper-1` and `gpt-4o-(mini-)transcribe` is *reported* for 2027-02-26 [42]. OpenAI's realtime transcription costs $0.017/min and is slow in independent tests: 0.69–0.81 s to the final transcript [34][1].
- **NVIDIA leads open *streaming* ASR.** Nemotron Speech Streaming (English) and Nemotron 3.5 ASR (40 locales, OpenMDW-1.1 license) use cache-aware chunks of 80–1120 ms [77]. NVIDIA's NeMo-Speech.cpp is a ggml runtime with Linux, macOS and Windows installers and an OpenAI-compatible server with realtime WebSocket [78]. Parakeet TDT 0.6B v3 (CC-BY-4.0, 25 languages) is the fastest accurate *offline* model. It runs at about 6,000× real time on GPU and 36× on a desktop CPU [12][89]. The most accurate open model on the Open ASR leaderboard is Qwen3-ASR-1.7B at 4.31 % average WER [12].
- **Cloud TTS:**
  - Cartesia Sonic 3.6 leads the AA Speech Arena (Elo 1273, $49 per 1M characters) [3].
  - Google's Gemini 3.8 Flash TTS (Elo 1260, $16.49/1M) and Flash-Lite TTS (1235, $11.03/1M) launched 2026-09-23 [3][56].
  - Inworld Realtime TTS-2 (1245) uses an OpenAI-Realtime-compatible WebSocket [68].
  - ElevenLabs Flash v2.5 (about 75 ms) is still the latency reference [30].
  - LMNT has shut down [70], and PlayHT's domain did not resolve when checked *(unverified status)*.
- **Local TTS:**
  - Kokoro-82M (Apache-2.0, AA Elo 1061) is statistically tied with NVIDIA Magpie-Multilingual 357M (1063, NVIDIA Open Model License) as the best-rated small open TTS, at a quarter of the size. It cannot stream text in [5][94][106].
  - Kyutai Pocket TTS (100M parameters, CC-BY-4.0) streams, clones voices, and gives about 200 ms to first audio on 2 CPU cores [93].
  - Best GPU picks: Chatterbox Turbo/Flash (MIT), Qwen3-TTS (Apache-2.0, 97 ms vendor claim) and CosyVoice 3 (Apache-2.0, 150 ms bi-streaming) [97][99][101][112].
  - Most top-rated open weights restrict use: Breeze TTS 2, Fish S2 Pro, Voxtral TTS and Higgs TTS 3 are non-commercial, and VibeVoice is research-only [108][102][107][109][100].
- **LLMs for voice.** Pipecat's voice-readiness benchmark (Aug 2026) is the most relevant evidence [10]. Within a ~700 ms time-to-first-answer-token budget, the best results are:
  - Qwen3.8-27B: 98.2 % at 649 ms p50 hosted, and 97.8 % at **101 ms** on a local RTX 5090.
  - Claude Haiku 4.5: 98.0 % at 637 ms.
  - Groq's gpt-oss-120b answers in 98 ms but passes only 86.3 %.
- **Fast inference.** AA measures gpt-oss-120b output speed at 1,734 tok/s on Cerebras, 710 on SambaNova, 473 on Groq and 88 on Together [7].
- **One client covers most LLMs.** Every major local server exposes an OpenAI-compatible API: Ollama, llama.cpp's `llama-server` (`--jinja` for tools), vLLM (Linux only), LM Studio and `mlx_lm.server` [73][74][75][76]. So do all the fast hosted providers. voice-agent-next needs an OpenAI-compatible client plus native Anthropic and Gemini adapters.
- **Packaging.** Target Python 3.11–3.13:
  - `onnxruntime` 1.30 and `websockets` 17 require Python ≥3.11. `kokoro` requires <3.13.
  - Native wheels for Windows, macOS and Linux exist for `sherpa-onnx`, `ctranslate2`, `pywhispercpp`, `moonshine-voice`, `piper-tts` and the Azure Speech SDK.
  - vLLM and SGLang ship Linux-only wheels [115].

---

## 2. How to read the numbers

- **AA-WER Streaming.** An Artificial Analysis composite of AA-AgentTalk (50 %), VoxPopuli (25 %) and Earnings22 (25 %), about 8 h of audio. Latency is the time from the end of speech (detected by SileroVAD) to the final transcript. If a model supports forced endpointing, the client forces it. Otherwise AA takes the first natural final within 2 s [1]. The "external endpoints" rows (Cartesia) therefore show latency when the client decides the turn has ended.
- **Pipecat TTFS and semantic WER.** 1,000 samples from `pipecat-ai/smart-turn-data-v3.1-train` run through an identical Pipecat pipeline. TTFS is the time from the user going silent to the final segment. Claude judges semantic WER against Gemini-generated references, and it ignores punctuation, contractions and fillers [9].
- **Open ASR average WER and RTFx.** Hugging Face's English short-form average. RTFx is seconds of audio processed per second of compute, on GPU in batch mode. The 2026 dataset mix adds "Voice Arena Monsoon" and private sets, so don't compare these numbers with 2025 figures [12].
- **TTS Elo.** Blind pairwise preferences from the AA Speech Arena [3] and from the community-run TTS Arena V2 [13].
- **TTFAT.** Time from the LLM request to the first answer token or tool call that the user would see; streamed reasoning doesn't count. Pipecat uses ~700 ms as the budget because the rest of an optimized pipeline adds about 500 ms [10].

---

## 3. Cloud streaming STT

### 3.1 Comparison (sorted roughly by independent accuracy)

| Provider / model | AA-WER streaming · final latency after end of speech [1] | Pipecat semantic WER · TTFS p50 / p95 [9] | List price | Languages | Turn detection / endpointing |
|---|---|---|---|---|---|
| xAI Grok Voice Transcribe 2.0 | **2.73 %** · 0.49 s | — | $3.33 per 1k min (AA) | multi | `endpointing` (400 ms default) plus an ML `smart_turn` threshold [66] |
| Meta Muse Voice Transcribe | 3.06 % · 0.16 s | **0.83 %** · 392 / 1,292 ms | $0.003/min [65] | 25+ | modes: push-to-talk, model endpointing, diarization (20+ speakers) [65] |
| Cartesia Ink-2 | 3.36 % · 0.43 s (semantic endpoints); 4.02 % · 0.067 s (external endpoints) | 1.25 % · 299 / 328 ms | $0.004/min (AA); 3 credits/s [52] | **English only** [51] | `turn.start`, `turn.eager_end`, `turn.end` [51] |
| ElevenLabs Scribe v2 Realtime | 3.59 % · 0.14 s | 3.12 % · 281 / 348 ms | $0.39/h = $0.0065/min [29] | 90+ | VAD or manual `commit` [31] |
| OpenAI `gpt-live-transcribe` | 3.92 % · 0.81 s | — | $0.017/min [34][35] | language hints | none server-side (`turn_detection: null`), so the client commits [37] |
| Google Gemini 3.5 Transcribe Live (preview) | 4.00 % · 0.40 s | 2.24 % · 458 / 532 ms | $9 per 1k min (AA) | 85+ [58] | Live API *(details unverified)* |
| AssemblyAI Universal-3.5 Pro Realtime | 4.02 % · 0.19 s (min-latency mode) | 1.22 % · 282 / 354 ms (U3.6 Pro: 0.96 % · 307 ms) | $0.45/h = $0.0075/min, billed on session time [24] | 18 [12][27] | neural end-of-turn: `min/max_turn_silence`, `vad_threshold`; `ForceEndpoint` [26] |
| Inworld STT 1 Realtime | 4.18 % · 0.07 s | — | $1.39 per 1k min (AA) | — | — |
| Speechmatics Agent STT (Linden-1) | 4.42 % · 0.16 s | 1.05 % · 369 / 438 ms | from $0.30/h, down to $0.16/h at volume [46] | 55+ [48] | `EndOfUtterance` (0–2 s trigger), conversational events [47][46] |
| Soniox `stt-rt-v5` | 4.50 % · **0.054 s** | 1.27 % · 260 / 305 ms | **$0.12/h = $0.002/min** [44] | 60+, translation included | semantic endpoint detection; `finalize` message [43][45] |
| Deepgram Nova-3 | 6.59 % · 0.066 s | 1.62 % · **247 / 298 ms** | $0.0048/min promo ($0.0077 list); multilingual $0.0058 ($0.0092) [15] | mono / multi | `endpointing` (10 ms default), `is_final` / `speech_final` [22] |
| Deepgram Flux | 7.39 % · **0.021 s** | — | $0.0065/min promo ($0.0077 list); multilingual $0.0078 [15] | EN; `flux-general-multi` covers 10 [17] | model-based EoT: `eot_threshold` 0.7, `eager_eot_threshold` 0.3–0.9, `eot_timeout_ms` 5,000 [16]; median EoT <300 ms, p95 1.5 s [23] |
| OpenAI `gpt-realtime-whisper` | 4.89 % · 0.69 s | 2.73 % · 740 / 878 ms | $0.017/min [34][36] | — | — |
| Mistral Voxtral Realtime (API) | 5.24 % · 0.68 s | 4.97 % · 525 / 973 ms | $0.006/min [63] | 13 | configurable delay, sub-200 ms to 2.4 s [63] |
| Azure Speech real-time | 5.25 % · 0.63 s | 1.18 % · 1,016 ms | $16.67 per 1k min (AA) | many | SDK events |
| Google Chirp 3 (Cloud STT v2) | 4.80 % · 1.28 s | — | $0.016/min for the first 500k min [59] | 85+ [59] | gRPC `StreamingRecognize` |
| Gladia Solaria-1 (live) | 7.82 % · 1.52 s | — | $0.75/h Starter → $0.25/h Growth [49] | 100+ | live mode supports `solaria-1` only [50] |
| NVIDIA Nemotron 3 ASR (open weights; also served by Together [71]) | 8.38 % @80 ms · 5.96 % @560 ms · 5.36 % @1,120 ms (latency 0.07–0.42 s) | 1.95 % · 221 / 238 ms | self-hosted | EN (3.5: 40 locales) | chunk size sets the latency [77] |
| Groq Whisper large-v3-turbo | batch HTTP only | — | $0.04/h [61] | 99 | none |

*Takeaways.*
- Flux, Ink-2 with external endpoints, Soniox and Nova-3 have the lowest finalization latency. AssemblyAI U3.5 and Speechmatics Linden give the best accuracy per unit of latency on Pipecat.
- Muse and Grok are the new accuracy leaders but have longer latency tails (Muse's p95 is 1.29 s).
- OpenAI, Azure, Chirp 3 and Gladia's live mode are too slow to finalize for tight turn-taking.

### 3.2 Protocol cheat-sheet

- **Deepgram Flux**
  - Endpoint: `wss://api.deepgram.com/v2/listen?model=flux-general-en|flux-general-multi` (add `language_hint` for the multilingual model).
  - Client sends binary audio; 80 ms chunks are "strongly recommended". Encodings: `linear16`, `linear32`, `mulaw`, `alaw`, `opus`, `ogg-opus` at 8–48 kHz.
  - Server events: `StartOfTurn`, `TurnInfo` (includes words), `EagerEndOfTurn`, `TurnResumed`, `EndOfTurn`, `Update` [16].
  - Nova-3 uses v1 `/listen` with `interim_results`, `is_final` and `speech_final` [22].
- **AssemblyAI v3**
  - Endpoint: `wss://streaming.assemblyai.com/v3/ws`, with US and EU variants. Pass `speech_model=universal-3-5-pro|universal-streaming-english|universal-streaming-multilingual`.
  - Encodings `pcm_s16le`, `pcm_mulaw`, `opus`, `ogg_opus`, `aac` at 8–96 kHz. Send binary chunks of 50–1000 ms, paced in real time.
  - Server messages: `Begin`, `Turn` (`end_of_turn`, `turn_is_formatted`, `words`), `SpeechStarted`, `Termination`.
  - Client messages: `UpdateConfiguration`, `ForceEndpoint`, `Terminate`. Keyterms prompt up to 100 terms [26].
- **ElevenLabs Scribe v2 Realtime**
  - Endpoint: `wss://api.elevenlabs.io/v1/speech-to-text/realtime`.
  - Audio goes in JSON messages (`input_audio_chunk`, base64); `commit_strategy=manual|vad`.
  - Server messages: `partial_transcript`, `committed_transcript`, `committed_transcript_with_timestamps`. Input is PCM 8–48 kHz or μ-law [31].
- **OpenAI realtime transcription**
  - A `type: "transcription"` session over WebSocket or WebRTC. Input is `audio/pcm` at **24 kHz**, sent with `input_audio_buffer.append` and closed with `.commit`.
  - Server events: `conversation.item.input_audio_transcription.delta` and `.completed`.
  - Options: `delay` from `minimal` to `xhigh`, plus `prompt`, `keywords` and `languages` [37][38].
- **Soniox**
  - Endpoint: `wss://stt-rt.soniox.com/transcribe-websocket`. The first message is a JSON config (`model`, `audio_format`, `sample_rate`, `language_hints`, `enable_endpoint_detection`, `enable_speaker_diarization`); binary audio follows.
  - Results arrive as `tokens` with `is_final`; send `{"type":"finalize"}` to force finalization [45].
- **Speechmatics**
  - Endpoint: `wss://{eu|global}.rt.speechmatics.com/v2`.
  - Client messages: `StartRecognition`, then binary `AddAudio`, then `EndOfStream`.
  - Server messages: `AddPartialTranscript`, `AddTranscript`, `EndOfUtterance`, `EndOfTranscript`.
  - `max_delay` ranges 0.7–4 s. Encodings `pcm_f32le`, `pcm_s16le`, `mulaw` [47].
- **xAI**
  - Endpoint: `wss://api.x.ai/v1/stt`. Configuration goes in query parameters; audio is binary PCM, μ-law, A-law or Opus at 8–48 kHz.
  - `transcript.partial` carries `is_final` and `speech_final`; `transcript.done` follows. Supports word timestamps [66].
- **Gladia.** `POST /v2/live` returns a WebSocket URL. Audio is 16 kHz, 16-bit PCM. Transcripts carry `is_final` [50].

### 3.3 2026 changes to track

- AssemblyAI released U3 Pro Streaming on 2026-03-03 [25] and U3.5 Pro Realtime on 2026-06-23. U3.5 is now the default `speech_model` and takes conversation context between turns [26][27]. A `universal-3-6-pro` model appears in Pipecat's table but not yet on the pricing page [9][24].
- Deepgram Flux Multilingual became generally available on 2026-04-29 [17]. Streaming prices for Nova-3 and Flux are currently promotional [15].
- Soniox's `stt-rt-v4` became an alias for `v5` after 2026-06-30 [43].
- ElevenLabs launched Scribe v2 Realtime on 2025-11-11 [28]. Cartesia launched Ink-2 on 2026-07-09 [51]. Meta launched Muse Voice Transcribe on 2026-09-03 [65]. Speechmatics launched Agent STT (Linden) on 2026-09-17 [46].
- Microsoft MAI-Transcribe-2 costs $0.10/h but is batch-only in preview *(from search result, not verified)* [60]. Google's Gemini 3.5 Transcribe arrived on 2026-08-26 and claims 70 % lower latency than Chirp 3 [58].

---

## 4. Local STT

### 4.1 Models (Open ASR English averages [12] unless noted)

| Model | Params | License | Languages | Streaming? | Avg WER | RTFx (GPU) |
|---|---|---|---|---|---|---|
| Qwen3-ASR-1.7B / 0.6B | 2.0B / 0.8B | Apache-2.0 | 52 (30 languages + 22 dialects) [90] | yes, with the vLLM backend [90] | **4.31** / 5.04 | 820 / 744 |
| NVIDIA Canary-Qwen-2.5B | 2.5B | CC-BY-4.0 | EN | no | 4.43 | 867 |
| IBM Granite Speech 4.1 2B (NAR) | 2B | Apache-2.0 | 6 (NAR: 5) | no | 4.62 (4.67) | 546 (2,074) |
| Cohere Transcribe 03-2026 | 2B | Apache-2.0 (gated) | 14 | no | 4.67 | 907 |
| **NVIDIA Parakeet TDT 0.6B v2 / v3** | 0.6B | CC-BY-4.0 | EN / 25 European (v3) [80] | offline (use with VAD) | 4.70 / 4.86 | **6,025 / 6,076** |
| Mistral Voxtral Small 24B | 24B | Apache-2.0 | 8 | no | 4.99 | 101 |
| **NVIDIA Nemotron Speech Streaming EN 0.6B** | 0.6B | NVIDIA Open Model License | EN | **yes, cache-aware**, 80/160/560/1120 ms chunks | 5.25 | 1,167 |
| **NVIDIA Nemotron 3.5 ASR streaming 0.6B** (2026-06-04) | 0.6B | OpenMDW-1.1 | 40 locales (32 usable without fine-tuning) [77] | yes, 80–1120 ms; `auto` language ID | 7.88 | 1,345 |
| Distil-Whisper large-v3.5 | 0.8B | MIT | EN | no | 5.40 | 879 |
| Kyutai `stt-2.6b-en` / `stt-1b-en_fr` | 2.6B / 1B | CC-BY-4.0 | EN / EN+FR | yes; 2.5 s / 0.5 s delay; the 1B has a semantic VAD [83] | 5.57 / — | 133 |
| Whisper large-v3 / large-v3-turbo | 2B / 0.8B (CSV) | Apache-2.0 / MIT (CSV) | 99 | no | 5.78 / 6.36 | 470 / 797 |
| Mistral Voxtral Mini 4B Realtime 2602 | 4B | Apache-2.0 | 13 | yes; delay configurable down to sub-200 ms [63] | 6.46 | 103 |
| Moonshine Streaming tiny / small / medium | 34M / 123M / 245M | MIT | EN (other languages in separate models) | **yes; built for 0.1–1 TOPS edge devices** [82] | 12.01 / 7.84 / 6.65 (model card) | — |
| NVIDIA Parakeet Realtime EOU 120M | 120M | NVIDIA Open Model License | EN, no punctuation | yes; 80–160 ms; emits an **end-of-utterance token** [79] | — | — |

### 4.2 Runtimes and Python packaging (PyPI and GitHub as of 2026-09-24 [115][116])

| Engine (latest) | What it runs | Streaming | Hardware / OS | Wheels | Speed evidence |
|---|---|---|---|---|---|
| `faster-whisper` 1.2.1 (2025-10-31) on `ctranslate2` 4.8.2 | Whisper, Distil-Whisper, turbo | VAD-chunked; batched | CPU int8/fp32 and CUDA | ctranslate2: Windows, macOS x86/arm64, Linux x86/aarch64 | 13 min of audio: large-v2 fp16 in 63 s (batched: 17 s) on an RTX 3070 Ti; `small` int8 in 102 s on an i7-12700K with 8 threads [85] |
| `whisper.cpp` v1.9.4 / `pywhispercpp` 1.5.1 | Whisper GGML | `whisper-stream` (500 ms step) plus Silero VAD | NEON/AVX, Metal, Core ML, CUDA, Vulkan, ROCm, OpenVINO [86] | Linux x86/aarch64, macOS arm64, Windows | `small` fp32: 125 s for 13 min on an i7-12700K [85] |
| `sherpa-onnx` 1.13.8 (2026-09-10) | streaming Zipformer/Paraformer; offline Parakeet, Whisper, Moonshine, SenseVoice; TTS; VAD | **yes, native** | Linux, macOS, Windows, Android, iOS, HarmonyOS, WASM [87] | every desktop platform, plus Android wheels | — |
| `onnx-asr` 0.12.0 | Parakeet v2/v3, Canary, Whisper | offline | ONNX Runtime CPU, CUDA, TensorRT, CoreML, DirectML | pure Python | Parakeet RTFx **36 on a Ryzen 9800X3D CPU**, 1.0 on a Cortex-A53, 57 on a T4, 320 on an RTX 5070 Ti with TensorRT [89] |
| **NeMo-Speech.cpp** (NVIDIA) | Nemotron 3.5/3 streaming, Parakeet TDT v3, CTC 1.1B, Sortformer diarization, Magpie TTS | yes; OpenAI-compatible HTTP plus realtime WebSocket server [78] | ggml on CPU, CUDA, Metal, Vulkan; Linux, macOS, Windows installers | native binaries, no PyPI package | one H100 serves 240 Nemotron 3.5 streams at 80 ms chunks and 2,400 at 1.12 s [77] |
| `nemo-toolkit` 3.0.0 | all NVIDIA checkpoints | cache-aware streaming scripts | PyTorch/CUDA, Linux-first | pure Python (heavy) | — |
| `moonshine-voice` 0.1.5 | Moonshine streaming | yes | CPU; Python, JS/WASM, iOS, Android, Raspberry Pi [81] | Linux x86/aarch64, macOS arm64, Windows | — |
| `moshi` 0.2.13 / `moshi-mlx` / Rust server | Kyutai STT and TTS | yes | CUDA, MLX, Rust | pure Python | 400 real-time streams on one H100; 64 connections at 3× real time on an L40S [84] |
| `parakeet-mlx` 0.5.2 / `mlx-audio` 0.5.5 / `mlx-whisper` 0.4.3 | Parakeet, Nemotron 3.5 streaming, Voxtral Realtime, Qwen3-ASR, Whisper, Moonshine [91][92] | partial | Apple Silicon | pure Python | — |
| `whisperx` 3.8.6 | Whisper plus alignment and diarization | batch-oriented *(unverified)* | Python 3.10–3.13 | pure Python | — |
| `vosk` 0.3.45 (**2022-12-14**) | Kaldi models | yes | CPU | Linux, Windows | stale; last release 2022 |

---

## 5. Cloud TTS

### 5.1 Comparison (AA Speech Arena Elo and price from [3]; list prices noted)

| Provider / model | AA Elo (rank of 92) | $ per 1M chars | Latency (vendor unless noted) | Text-in streaming | Timestamps / barge-in data | Output formats | Languages |
|---|---|---|---|---|---|---|---|
| **Cartesia Sonic 3.6** (`sonic-3.6-2026-08-27`) | **1273 (#1)** | 49 | sub-90 ms model latency [55] | yes: WebSocket `context_id` plus `continue`, `max_buffer_delay_ms` | **word and phoneme** timestamps [54] | raw pcm f32/s16, μ-law, A-law at 8–48 kHz [54] | 44 [53] |
| Google Gemini 3.8 Flash TTS / Flash-Lite TTS (2026-09-23) | 1260 (#2) / 1235 (#6) | 16.49 / 11.03 | not published | audio-out streaming only *(text-in streaming unverified)* | — | L16 PCM 24 kHz; μ-law/A-law at 8/16/24 kHz [57] | 130 / 101 [57] |
| Alibaba Qwen-Audio-3.0-TTS-Plus | 1259 (#3) | 27.59 | — | — | — | — | — |
| Inworld Realtime TTS-2 / TTS-2 Flash | 1245 (#4) / 1210 (#8) | 20.83 / 10.42 (on-demand list $25 / $15 [120]) | sub-100 ms TTFB [68] | yes, OpenAI-Realtime-compatible WebSocket [68] | — | 16 / 48 kHz examples [68] | 200+ (vendor) |
| Speechify Simba 3.2 | 1237 (#5) | **6.58** | — | — | — | — | — |
| ElevenLabs v3 Conversational | 1196 (#12) | 50 | ~280 ms [30] | yes: `stream-input` and multi-context WebSockets [32][33] | character-level alignment [32] | PCM 16/22.05/24/44.1 kHz, `ulaw_8000`, MP3, Opus [32] | 70+ |
| ElevenLabs Flash v2.5 | 1074 (#44) | 50 ($0.05 per 1K [29]) | **~75 ms** [30] | same | same | same | 32 |
| Smallest.ai Lightning v3.1 Pro | 1175 (#14) | 19.5 | sub-100 ms [121] | streaming API | — | 44.1 kHz example | 70+ (vendor) |
| Soniox TTS Real-Time v2 | 1173 (#15) | 14.23 (~$0.70/h [44]) | — | — | — | — | — |
| MiniMax Speech 2.8 HD / Turbo | 1168 / 1149 | 100 / 60 | — | — | — | — | — |
| Murf Falcon 2 | 1157 | 10 | — | — | — | — | — |
| Fish Audio S2.1 Pro (API) | 1139 | 15 | — | — | — | — | 80+ |
| Azure HD 2.5 | 1128 | 22 | — | — | — | — | — |
| Deepgram Aura-2 | not rated | 30 ($0.030 per 1K [15]) | *(unverified)* | WebSocket `Speak` / `Flush` / `Clear` / `Close` [20] | — | linear16, μ-law, A-law at 8–48 kHz [20] | 7 languages, 87 voices [21] |
| Deepgram Flux TTS (Aug 2026) | not rated | 45 ($0.045 per 1K [15]) | as low as 80 ms [18] | WebSocket `/v2/speak`; conversation-aware | `text_spoken` / `text_remaining` returned on interruption [19] | *(unverified)* | EN [18] |
| Rime Arcana v3 | 1000 (#74) | 50 | ~200 ms cloud, 120 ms on-prem [69] | WebSocket | word timestamps [69] | *(unverified)* | 10 |
| Hume Octave 2 | 1049 (#57) | 87.5 (AA); $0.05–0.15 per 1K by plan [67] | — | — | — | — | — |
| OpenAI `gpt-4o-mini-tts` | not listed (TTS-1 HD: 1099) | $0.60/1M text tokens in, $12/1M audio tokens out [41] | use `wav`/`pcm` for speed [40] | **no**: HTTP chunked audio only | none | MP3, Opus, AAC, FLAC, WAV, PCM 24 kHz [40] | ~99 [40] |
| Groq Orpheus V1 English | — | 22 [61] | — | — | — | — | EN |
| Mistral Voxtral TTS (API) | 1076 | 16 [64] | 70 ms model latency [64] | — | — | — | 9 |
| LMNT | — | — | **shut down** [70] | | | | |

### 5.2 Notes

- **Text-in streaming and barge-in.** Barge-in means the user interrupts while the agent is talking. The agent must then truncate its transcript at what was actually spoken. Only Cartesia (word and phoneme timestamps), ElevenLabs (character alignment), Rime (word timestamps) and Deepgram Flux TTS (spoken vs. remaining text) return that information natively. For OpenAI, Gemini and Aura-2, voice-agent-next must estimate the cut point from audio duration.
- **Multi-context WebSockets.** Cartesia's `context_id` and ElevenLabs' `multi-stream-input` let one socket carry overlapping utterances and cancel them cleanly [54][33].
- **The TTS Arena V2 community ranking differs from AA.** Its top entries are Aurora (stealth, 1580), CastleFlow v1.0 (1559), Inworld TTS MAX (1558), Deepdub eTTS 3.2, Inworld TTS, Papla P1, Lightning v3.1 Pro, Hume Octave and MiniMax Speech 2.8 HD. The best open model there is Kokoro v1.0 at #30 [13].

---

## 6. Local TTS

| Model | Params | License | Languages | Streaming | Latency / speed evidence | Voice cloning | Runtime / package |
|---|---|---|---|---|---|---|---|
| **Kokoro-82M v1.0** | 82M | Apache-2.0 | 8 languages, 54 voices [94] | sentence-level | AA Elo 1061, #6 among open models [5]; RTF 3.19 on a Raspberry Pi 4B with 4 threads, i.e. slower than real time [88]; "near real-time on M1" [95] | no | `kokoro` 0.9.4 (torch, Python <3.13), `kokoro-onnx` 0.6.1 (~300 MB, ~80 MB quantized), `sherpa-onnx`, `mlx-audio`; 24 kHz |
| **Kyutai Pocket TTS** | 100M | CC-BY-4.0 (voices vary) | EN, FR, DE, PT, IT, ES | **audio streaming** | ~200 ms to first chunk; ~6× real time on an M4 Air CPU using 2 cores [93] | yes (WAV prompt) | `pocket-tts` 3.2.0 (Python 3.10–3.14, CPU torch); also `sherpa-onnx`, ONNX, Rust and WASM ports |
| Piper (`piper1-gpl` 1.8.0) | small VITS | **GPL-3.0** (embeds espeak-ng) [96] | many voices | sentence-level | fast on CPU *(unverified figures)* | no | `piper-tts` wheels for every desktop platform |
| Soprano-1.1-80M | 80M | Apache-2.0 | EN | yes ("lossless streaming") | <15 ms on GPU, <250 ms on CPU; 2000× / 20× real time (vendor); 32 kHz [103] | no | `soprano-tts` (wheel is CUDA-only; install from source for CPU/MPS) |
| Chatterbox-Nano | 110M | MIT | EN | chunked | 3× real time on an 8-core CPU (vendor) [98] | yes | `chatterbox-tts` 0.1.7 |
| **Chatterbox-Turbo** | 350M | MIT | EN; paralinguistic tags | chunked | 1-step mel decoder; hosted version sub-200 ms (vendor) [97] | yes | `chatterbox-tts`; ONNX export available |
| Chatterbox-Flash | 0.5B | MIT | EN | native block streaming | time to first packet 103–118 ms, RTF 0.076–0.107 (FlashInfer + CUDA graphs) [99] | yes | `chatterbox-flash` |
| Chatterbox-Multilingual | 500M | MIT | 23 [97] | chunked | — | yes | `chatterbox-tts` |
| NeuTTS Air / Nano | 748M / 229M | Apache-2.0 (gated) / other | EN (Nano: DE, ES, FR variants) | — | GGUF Q4/Q8, "real-time on mid-range devices" (vendor) [104] | yes, from 3 s of audio | `neutts` 1.4.1 (Linux x86, macOS arm64, Windows wheels) plus `llama-cpp-python` |
| **Qwen3-TTS** 0.6B / 1.7B | ~0.9B / 1.7B | Apache-2.0 | 10 | yes | 97 ms end-to-end (vendor) [101] | yes, from 3 s | `qwen-tts` 0.1.1; `mlx-audio` |
| **Fun-CosyVoice3-0.5B** | 0.5B | Apache-2.0 | 9 plus 18 Chinese dialects | **bi-streaming** (text in and audio out) | as low as 150 ms (vendor) [112] | yes | CosyVoice repo (GPU) |
| VibeVoice-Realtime-0.5B | 0.5B LM (~1.0B total) | MIT, but stated as **research use** | EN (+9 experimental) | streaming text input | ~300 ms to first audible audio (vendor) [100] | single speaker | transformers; WebSocket demo |
| Kyutai TTS `tts-1.6b-en_fr` | 1.6B | CC-BY-4.0 | EN, FR | streaming text input [84] | Rust server *(figures unverified)* | voice repo | `moshi`, Rust, MLX |
| Orpheus 3B | 3.8B | Apache-2.0 | EN (+ multilingual research release) | yes | ~200 ms, ~100 ms with input streaming (vendor) [111] | limited | `orpheus-speech` (vLLM) |
| KaniTTS 370M | 370M | LFM Open License v1.0 in metadata; card badge says Apache-2.0 | 6 | — | ~1 s to generate 15 s of audio on an RTX 5080; 2 GB VRAM [105] | fixed speakers | `kani-tts` |
| Sesame CSM-1B / Dia2-2B / Maya1 | 1.55B / 1.9B / 3.3B | Apache-2.0 [114][110] | EN | Dia2: streaming text input (server "upcoming") [110] | Dia2 needs CUDA 12.8+ | context / prefix conditioning | transformers, `dia2` |
| NVIDIA Magpie TTS Multilingual | 357M | NVIDIA Open Model License | 12 | — | 22.05 kHz; no zero-shot cloning [106] | no | NeMo, NeMo-Speech.cpp (GGUF) [78] |
| Voxtral-4B-TTS-2603 | 4B | **CC-BY-NC-4.0** | 9 | yes | ~70 ms model latency (API) [64] | 5–25 s prompt | vLLM, `mlx-audio` [107] |
| Fish Audio S2 Pro | 4.4B + 0.4B | **Fish Audio Research License (non-commercial)** | 80+ | yes (SGLang) | time to first audio ~100 ms, RTF 0.195 on an H200 [102] | yes | `fish-speech` |
| Breeze TTS 2 | 3.5B | **non-commercial weights** [108] | EN/ZH (50 claimed) | yes | <40 ms time to first audio on an H100 (secondhand) [108] | yes, plus voice design | `breeze-tts` |
| Higgs TTS 3 4B | 4B | **research / non-commercial** [109] | 70+ | — | — | — | — |
| F5-TTS / Spark-TTS / IndexTTS-2 | — | CC-BY-NC-4.0 / CC-BY-NC-SA-4.0 / bilibili Model Use License [114][113] | EN/ZH | no / — / — | — | yes | — |
| Zonos v0.1 / MeloTTS / Supertonic-2 | 1.6B / small / — | Apache-2.0 / MIT / OpenRAIL [114] | — | — | *(unverified)* | Zonos: yes | `melotts` (last sdist 2024) |

**Takeaways.**
- On CPU, the realistic choices are Kokoro (quality), Pocket TTS (streaming and cloning), Piper (tiny, but GPL) and, as experiments, Soprano and Chatterbox-Nano. Kokoro on a Raspberry Pi 4 runs slower than real time [88].
- For 16 GB GPUs, the permissive, low-latency choices are Chatterbox-Turbo/Flash, Qwen3-TTS and CosyVoice 3.
- Of AA's top five open-weight TTS models, three are non-commercial (Breeze TTS 2, Fish S2 Pro, Voxtral TTS). Magpie (#5) uses the NVIDIA Open Model License, and Step Audio EditX (#3) has no license tag on Hugging Face *(unverified)* [5][108][102][107][106].

---

## 7. LLMs for voice

### 7.1 Voice-specific evidence (Pipecat Benchmark 01, Aug 18 2026: 46 configs, 30-turn scripted tool-use conversations; pass rate is strict per turn [10])

| Config | Where it ran | Pass rate | TTFAT p50 / p95 |
|---|---|---|---|
| Qwen3.8-27B, thinking off, FP8 | Baseten | **98.2 %** | 649 / 801 ms |
| Claude Haiku 4.5 | Anthropic | 98.0 % | 637 / 1,615 ms |
| Qwen3.8-27B, thinking off, NVFP4 | **local RTX 5090** | 97.8 % | **101 / 318 ms** |
| Qwen3.6-27B, thinking off | Baseten | 97.3 % | 667 / 769 ms |
| Gemini 3.6 Flash (minimal) | AI Studio | 97.1 % | 798 / 984 ms |
| Gemma 4 31B-it, thinking off | Baseten | 96.6 % | 489 / 609 ms |
| GPT-4.1 | OpenAI | 96.3 % | 536 / 1,771 ms |
| Claude Sonnet 4.6 (off the clock) | Anthropic | 100 % | 850 / 4,126 ms |
| GPT-5.6 Luna (none) | OpenAI | 88.3 % | 671 / 2,304 ms |
| gpt-oss-120b | **Groq** | 86.3 % | **98 / 217 ms** |
| GPT-4o-mini / GPT-5-mini | OpenAI | 82.7 % / 83.7 % | 553 / 682 ms |
| Gemma 4 26B-A4B, thinking off | Baseten | 80.7 % | 578 / 634 ms |
| Gemini 3.5 Flash-Lite (minimal) | AI Studio | 68.6 % | 591 / 679 ms |
| Nemotron 3.5 Lightning, thinking off | local RTX 5090 | 50.9 % | 62 / 70 ms |

**PhoneBench Alpha 1** (Pipecat, Aug 27 2026; 15 models on phone-assistant dialogue and tool calls) [11] ranks:

| Model | Score | TTFAT p50 | Cost per minute |
|---|---|---|---|
| Gemini 3.6 Flash | 78.6 % | 1,168 ms | $0.075 |
| GPT-5.6 Terra | 72.4 % | 980 ms | $0.035 |
| PhoneLLM 30B (open) | 72.3 % | 331 ms | $0.0025 |
| GPT-5.6 Luna | 70.7 % | 786 ms | $0.0035 |
| Qwen 3.8 27B | 70.0 % | — | $0.0074 |
| Claude Haiku 4.5 | 67.8 % | 707 ms | $0.019 |
| Gemma 4 31B | 58.1 % | 385 ms | $0.010 |

On **BFCL V4** (last updated 2026-04-12), Claude Haiku 4.5 is #6 overall at 68.70 % with a 1.68 s mean latency. GLM-4.6 is the best open model at 72.38 % [14].

### 7.2 Fast inference providers (AA, gpt-oss-120B with high reasoning [7]; vendor figures noted)

| Provider | Output speed (AA) | Time to first token, excluding reasoning (AA) | Vendor claims / price |
|---|---|---|---|
| Cerebras | **1,734 tok/s** | 0.48 s | ~3,000 tok/s on gpt-oss-120b and ~1,850 tok/s on `qwen-3.8-27b` [62] |
| SambaNova | 710 tok/s | 1.07 s | — |
| Groq | 473 tok/s (gpt-oss-20b: 878) | 0.73 s (20b: 0.81 s) | 120b: $0.15 in / $0.60 out per 1M, ~500 tps; 20b: $0.075 / $0.30, ~1,000 tps; Llama 3.1 8B at 560 tps [61][8] |
| Baseten / Crusoe / DeepInfra Turbo | 200 / 260 / 253 tok/s | **0.27 / 0.36 / 0.64 s** | — |
| Together AI | 88 tok/s | 0.55 s | blended $0.195 per 1M [7] |
| Google Vertex (gpt-oss-20b) | 374 tok/s | 0.33 s | — [8] |

Fireworks was not in AA's displayed rows for this model, and its docs index no longer lists ASR endpoints *(unverified)*.

### 7.3 Fast frontier models (standard-tier list price per 1M tokens, input / output)

| Model | Price | Voice evidence |
|---|---|---|
| OpenAI `gpt-6-luna` (Sep 2026) | $0.10 / $0.50 [34] | AA measures 141 tok/s [6]; not yet in voice benchmarks [119] |
| OpenAI `gpt-5.6-luna` / `gpt-5.4-nano` / `gpt-5.4-mini` | $0.20 / $1.20 · $0.20 / $1.25 · $0.75 / $4.50 [34] | 5.6 Luna: 88.3 % at 671 ms [10] |
| OpenAI `gpt-4.1` / `gpt-4.1-mini` | $2 / $8 · $0.40 / $1.60 [34] | 96.3 % at 536 ms · 85.3 % at 851 ms [10] |
| Google `gemini-3.8-flash` | $0.75 / $3.75 until 2026-12-31, then $1.50 / $7.50 [56] | AA measures 297 tok/s [6] |
| Google `gemini-3.5-flash-lite` | $0.30 / $2.50 [56] | fastest on AA at 351 tok/s [6], but only 68.6 % pass [10] |
| Anthropic `claude-haiku-4-5` | $1 / $5; "fastest" Claude model [72] | 98.0 % at 637 ms [10]; #6 on BFCL V4 [14] |
| Anthropic `claude-sonnet-5` | $2 / $10 [72] | 93.0 % at 1,204 ms [10] |

### 7.4 Local serving (all OpenAI-compatible)

| Server | Endpoints | Tool calling | Platforms / default port | Latest release [116] |
|---|---|---|---|---|
| Ollama | `/v1/chat/completions`, `/completions`, `/models`, `/embeddings`, `/responses` | yes, including thinking control [73] | Windows / macOS / Linux, `:11434` | v0.34.4 (2026-09-23) |
| llama.cpp `llama-server` | `/v1/chat/completions`, `/completions`, `/responses`, `/embeddings`; prompt caching, speculative decoding, parallel slots | with `--jinja` [74] | all platforms, `:8080` | v0.5.0 (2026-09-23) |
| vLLM 0.30.0 | OpenAI-compatible server | yes (tool parsers) | **Linux-only wheels** [115] | 2026-09-22 |
| LM Studio | `/v1/models`, `/responses`, `/chat/completions`, `/embeddings`, `/completions` [75] | yes *(unverified details)* | desktop app, `:1234` | — |
| `mlx_lm.server` 0.31.3 | OpenAI-like chat API; "not recommended for production" [76] | *(unverified)* | Apple Silicon, `:8080` | 2026-04-22 |

### 7.5 Small local models for tool calling

| Model | Size | License | Evidence |
|---|---|---|---|
| Qwen3.8-27B (Aug 2026) | 27.8B dense | Apache-2.0 | best voice-readiness result: 98.2 %; 97.8 % at 101 ms on an RTX 5090 [10] |
| Qwen3.6-27B / Qwen3.6-35B-A3B | 27B / 36B total, 3B active | Apache-2.0 | 97.3 % at 667 ms / 91.6 % at 764 ms [10] |
| Qwen3.5-9B / 4B | 9.7B / 4.7B | Apache-2.0 | vendor BFCL-V4 66.1 / 50.3; TAU2 79.1 / 79.9 [117] |
| Gemma 4 31B / 26B-A4B / 12B / E4B / E2B | 31B–2B | Apache-2.0 | vendor Tau2 76.9 / 68.2 / 69.0 / 42.2 / 24.5 % [118]; 31B scores 96.6 % at 489 ms [10] |
| gpt-oss-20b | 21B MoE | Apache-2.0 | 878 tok/s on Groq [8]; fits 16 GB *(unverified)* |
| Ministral 3 (3B / 8B / 14B), Granite 4.2 (3B / 8B / 30B) | — | Apache-2.0 | "native function calling" and "reasoning-augmented tool calling" (vendor cards) |
| Previous generation on BFCL V4 | — | Apache-2.0 | Qwen3-8B 42.6 %, Qwen3-4B-2507 35.7 %, Qwen3-1.7B 28.4 % [14] |

---

## 8. Leaderboard snapshot (2026-09-24)

- **AA streaming STT (AA-WER):** Grok Voice Transcribe 2.0 2.73 %, Muse 3.06 %, Cartesia Ink Preview 3.11 %, Ink-2 3.36 %, Scribe v2 Realtime 3.59 %, Qwen3 ASR Flash Realtime 3.73 %, GPT Live Transcribe 3.92 % [1].
- **AA batch STT (AA-WER v2):** StepAudio 3 ASR 1.73 %, Fun-Realtime-ASR-preview 1.73 %, MAI-Transcribe-2 2.04 %, Scribe v2 2.18 %, Grok Voice Transcribe 2.0 2.29 %. The best open-weights model is Voxtral Small at 2.77 % [2].
- **Open ASR, English (2026-09-19 build):**
  - Proprietary: Zoom Scribe v2 Pro 3.59, Azure Speech 07-2026 3.81, Modulate 3.84, ElevenLabs Scribe v2 3.97.
  - Open: Qwen3-ASR-1.7B 4.31, Hojo-ASR-V1 4.33, Higgs Audio V3 STT 4.39, Canary-Qwen-2.5B 4.43.
  - Throughput leaders: Granite Speech 5.0 470M TurboCTC (RTFx 12,946 at 5.04) and Parakeet TDT v2/v3 (RTFx ~6,000) [12].
- **AA TTS arena:** Sonic 3.6 1273, Gemini 3.8 Flash TTS 1260, Qwen-Audio-3.0-TTS-Plus 1259, Inworld Realtime TTS-2 1245, Simba 3.2 1237. Top open models: Breeze TTS 2 1204, Fish S2 Pro 1120, Step Audio EditX 1094, Voxtral TTS 1076, Magpie 1063, Kokoro 1061 [3][5].
- **AA LLM Intelligence Index v4.3.2:** Claude Opus 5.5 57.6, Claude Fable 5.1 53.4, GPT-6 Astra 52.7 [6]. These are too slow for a voice turn. Use §7.1 instead.

---

## 9. Implications for voice-agent-next

### 9.1 Provider priority

| Priority | STT | TTS | LLM |
|---|---|---|---|
| **P0 (MVP)** | Cloud: Deepgram (Nova-3 and Flux), AssemblyAI (U3.5 Pro RT and Universal-Streaming), Soniox v5, OpenAI realtime transcription. Local: `sherpa-onnx` (Parakeet v3, streaming Zipformer, Moonshine), `faster-whisper`, `moonshine-voice` | Cloud: Cartesia Sonic 3.6, ElevenLabs Flash v2.5 / v3 Conversational, OpenAI `gpt-4o-mini-tts`, Deepgram Aura-2. Local: Kokoro (`kokoro-onnx`), Pocket TTS | Generic OpenAI-compatible client (OpenAI, Groq, Cerebras, SambaNova, Together, Ollama, llama.cpp, vLLM, LM Studio, mlx-lm); native Anthropic |
| **P1** | Cloud: Speechmatics Linden, Cartesia Ink-2, ElevenLabs Scribe v2 RT, Google (Gemini 3.5 Transcribe Live, Chirp 3), Azure, Meta Muse, xAI, Mistral Voxtral RT, Groq Whisper (batch fallback). Local: NeMo-Speech.cpp (Nemotron streaming) via its local server, `parakeet-mlx` / `mlx-audio`, `whisper.cpp`, Kyutai STT, Voxtral Realtime / Qwen3-ASR via vLLM | Cloud: Inworld TTS-2, Gemini 3.8 TTS, Rime Arcana v3, Deepgram Flux TTS, Azure, Hume, Soniox TTS, Groq Orpheus. Local: Chatterbox Turbo/Nano/Multilingual, Qwen3-TTS, CosyVoice 3, NeuTTS, Soprano, Piper (optional extra because of GPL) | Native Gemini (`google-genai`) |
| **P2** | Gladia (slow live finals), WhisperX (evaluation), Canary-Qwen / Granite / Cohere (batch accuracy), Vosk (legacy) | Orpheus, CSM, Dia2, VibeVoice-Realtime (research-only), Magpie (NVIDIA OML), Maya1, KaniTTS, Zonos, MeloTTS; non-commercial models behind an explicit `license_ack` flag (Fish S2, Voxtral TTS, Breeze, Higgs, F5, Spark, IndexTTS) | — |
| **Exclude** | — | LMNT (shut down), PlayHT (offline *unverified*) | — |

### 9.2 Recommended default stacks

| Profile | STT (and turn detection) | LLM | TTS | Rationale |
|---|---|---|---|---|
| **Local CPU** (x86 or ARM laptop, no GPU) | English: Moonshine Streaming small/medium (`moonshine-voice`, MIT). Multilingual: Parakeet TDT v3 int8 on VAD-cut segments via `sherpa-onnx` / `onnx-asr` | Qwen3.5-4B or Gemma 4 E4B, Q4 GGUF, via llama.cpp or Ollama; offer a cloud LLM in hybrid mode | Kokoro-82M (`kokoro-onnx`); Pocket TTS when streaming or cloning is needed | Parakeet runs 36× real time on a desktop CPU [89]; Moonshine targets edge devices [82]; Pocket TTS gets first audio in about 200 ms on 2 cores [93]. CPU contention among STT, LLM and TTS is the main risk, so the benchmark suite must measure it. |
| **Local GPU, 16 GB NVIDIA** | Nemotron Speech Streaming EN (160–560 ms chunks) or Nemotron 3.5 ASR through the NeMo-Speech.cpp server; `faster-whisper` large-v3-turbo for 99-language fallback | Qwen3.5-9B, Gemma 4 12B or gpt-oss-20b via `llama-server` (vLLM on Linux) with thinking off | Chatterbox-Turbo (English, MIT) or Qwen3-TTS-0.6B / CosyVoice 3 (multilingual, Apache-2.0); Kokoro as fallback | Cache-aware streaming gives 221 ms TTFS [9]; the Qwen3.8-27B local result [10] shows a local LLM can beat cloud TTFAT. Hardware with 24–32 GB can run Qwen3.8-27B NVFP4. |
| **Apple Silicon** (≥16 GB) | `parakeet-mlx` (v3) or `mlx-audio` Nemotron 3.5 streaming; Kyutai STT via `moshi-mlx` (EN/FR, semantic VAD); `whisper.cpp` with Metal/Core ML for other languages | `mlx_lm.server`, LM Studio (MLX engine) or Ollama with Qwen3.5-9B or Gemma 4 E4B/12B at 4-bit | Kokoro or Qwen3-TTS via `mlx-audio`; Pocket TTS on CPU | `mlx-audio` covers most models on one framework [91]. |
| **Cloud, low latency** | Deepgram Flux (English; EoT built in) or AssemblyAI U3.5 Pro RT; Soniox v5 for multilingual or lowest cost | Claude Haiku 4.5, GPT-4.1 or hosted Qwen3.8-27B for quality; Groq or Cerebras gpt-oss-120b when speed matters most | Cartesia Sonic 3.6 (WebSocket continuations and word timestamps) or ElevenLabs Flash v2.5; Inworld TTS-2 Flash for low cost | These are the fastest finals [1], the best pass rates inside 700 ms [10], and the top arena TTS [3]. |

### 9.3 Integration approach per provider (PyPI versions as of 2026-09-24 [115])

| Provider | Approach | Package |
|---|---|---|
| Deepgram (STT v1/v2, TTS v1/v2) | **Raw WebSocket** via `websockets` 17.1: small JSON event set, and it avoids depending on SDK major-version changes | `deepgram-sdk` 7.10.0 (optional) |
| AssemblyAI, Soniox, xAI, Gladia, Meta Muse, Rime | **Raw WebSocket** (Gladia: HTTP init first) | `assemblyai` 1.5.5 and `soniox` 2.9.0 exist but aren't needed |
| ElevenLabs | Raw WebSocket for realtime STT and for TTS `stream-input` / multi-context; `httpx` for REST | `elevenlabs` 2.69.0 |
| Cartesia | Official SDK or raw WebSocket (versioned `cartesia_version` parameter) | `cartesia` 4.2.0 (Apache-2.0) |
| Speechmatics | Official realtime SDK (larger message set) | `speechmatics-rt` 1.1.1 |
| OpenAI | SDK for HTTP TTS and batch; raw WebSocket or WebRTC for realtime transcription | `openai` 3.19.2 |
| Inworld | Reuse the OpenAI-Realtime protocol client [68] | — |
| Google | Gemini Live, TTS and Transcribe through `google-genai`; Chirp 3 over gRPC | `google-genai` 2.25.0; `google-cloud-speech` 2.40.0 |
| Azure | **SDK required** (proprietary protocol); native wheels for Windows, macOS and Linux | `azure-cognitiveservices-speech` 1.51.2 |
| Hume, Mistral | Official SDKs | `hume` 0.14.1; `mistralai` 2.10.1 |
| Fast LLM providers and local servers | `openai` client with a custom `base_url` | vendor SDKs (`groq` 1.7.0, `cerebras-cloud-sdk` 1.91.0, `sambanova` 1.13.0, `together` 2.36.0, `fireworks-ai` 1.2.15) are optional |
| Anthropic | Native SDK | `anthropic` 1.8.0 |
| NeMo-Speech.cpp, Pocket TTS server, Kyutai Rust server | Talk to the local server over HTTP or WebSocket | binaries or `pocket-tts` |

### 9.4 Design requirements this implies

1. **STT event model**:
   - Events: partial, final, and `turn_started` / `eager_end_of_turn` / `turn_resumed` / `end_of_turn`, plus confidence, word timings and detected language.
   - A provider-agnostic `force_finalize()` that maps to AssemblyAI `ForceEndpoint`, Soniox `finalize`, OpenAI `input_audio_buffer.commit` and ElevenLabs manual `commit` [26][45][37][31]. For Speechmatics, a per-turn force message is *unverified*; tune `max_delay` and `end_of_utterance_silence_trigger` instead [47]. Forced finalization is how "external endpoints" reach finals in 67 ms [1].
2. **Audio plumbing**:
   - Resample per provider: 16 kHz for most STT, **24 kHz for OpenAI realtime** [37], 8 kHz μ-law for telephony.
   - Pace chunks at real time: 80 ms for Flux, 50–1000 ms for AssemblyAI [16][26].
   - Some protocols need base64 JSON framing (ElevenLabs STT).
3. **TTS contract**:
   - `push_text(delta)`, `flush()`, `cancel(context)` and optional word timestamps.
   - Fall back to sentence chunking when a provider has no text-in streaming: OpenAI, Kokoro and Piper, and Gemini *(unverified)*.
   - Declare each provider's native output sample rate: 22.05 kHz for Magpie and Kani, 24 kHz for Kokoro, OpenAI and Gemini, 32 kHz for Soprano, up to 48 kHz for Cartesia and ElevenLabs.
4. **Model registry**. Store pricing units (per minute, per session hour for AssemblyAI [24], per character, or per token for Soniox, OpenAI TTS and Gemini TTS), license class (permissive / non-commercial / GPL / research), dated model IDs (for example `sonic-3.6-2026-08-27`), and deprecation dates. Providers disappear and model IDs get aliased (LMNT, Soniox v4→v5, OpenAI 2027).
5. **Packaging**:
   - Support Python 3.11–3.13.
   - Put heavy local engines behind extras (`[whisper]`, `[sherpa]`, `[nemo]`, `[mlx]`, `[kokoro]`, `[piper-gpl]`).
   - Mark `vllm` and NeMo as Linux-only and MLX as macOS-only.

### 9.5 What the benchmark suite should measure

- **STT:**
  - Pipecat-style TTFS p50/p95/p99 and semantic WER plus raw WER [9], using `pipecat-ai/smart-turn-data-v3.1-train` for comparability.
  - AA-style latency from VAD end of speech, with and without forced finalization [1].
  - End-of-turn latency and false end-of-turn rate for STT with built-in turn detection.
- **TTS:** TTFB (time to first byte), RTF, characters per second, and pronunciation robustness on numbers, codes and terms. AA scores these as separate categories [4].
- **LLM:** TTFAT p50/p95 and strict multi-turn pass rate with tools, following Pipecat's aiwf harness and its 700 ms guideline [10], plus tokens per second.
- **End to end:** voice-to-voice latency per stack profile in §9.2, with CPU, GPU and RAM contention recorded, because local stacks share one machine.

---

## 10. Sources

**Leaderboards and benchmarks**
1. Artificial Analysis — Speech to Text streaming leaderboard (embedded data read 2026-09-24). https://artificialanalysis.ai/speech-to-text/streaming
2. Artificial Analysis — Speech to Text (batch) leaderboard (2026-09-24). https://artificialanalysis.ai/speech-to-text
3. Artificial Analysis — Text to Speech leaderboard, provider voices (2026-09-24). https://artificialanalysis.ai/text-to-speech/leaderboard
4. Artificial Analysis — Text to Speech overview: pronunciation, price, speed (2026-09-24). https://artificialanalysis.ai/text-to-speech
5. Artificial Analysis — Open-weights TTS leaderboard (2026-09-24). https://artificialanalysis.ai/text-to-speech/leaderboard/provider-voice/open-weights
6. Artificial Analysis — LLM models, intelligence and speed (2026-09-24). https://artificialanalysis.ai/models
7. Artificial Analysis — gpt-oss-120B provider benchmarks (2026-09-24). https://artificialanalysis.ai/models/gpt-oss-120b/providers
8. Artificial Analysis — gpt-oss-20B provider benchmarks (2026-09-24). https://artificialanalysis.ai/models/gpt-oss-20b/providers
9. Pipecat — stt-benchmark README results table (Sep 2026). https://github.com/pipecat-ai/stt-benchmark
10. Pipecat — Benchmark 01: Voice readiness (2026-08-18). https://www.pipecat.ai/benchmarks/voice-readiness
11. Pipecat — Benchmark 03: PhoneBench Alpha 1 (2026-08-27). https://www.pipecat.ai/benchmarks/phonebench-alpha-1
12. Hugging Face — Open ASR Leaderboard and results dataset (`english_short_latest.csv`, updated 2026-09-22). https://huggingface.co/spaces/hf-audio/open_asr_leaderboard · https://huggingface.co/datasets/hf-audio/open-asr-leaderboard-results
13. TTS-AGI — TTS Arena V2 (leaderboard API, 2026-09-24). https://huggingface.co/spaces/TTS-AGI/TTS-Arena-V2
14. Berkeley Function Calling Leaderboard V4 (last updated 2026-04-12). https://gorilla.cs.berkeley.edu/leaderboard.html

**Cloud STT, TTS and LLM providers**
15. Deepgram — Pricing. https://deepgram.com/pricing
16. Deepgram — Getting Started with Flux. https://developers.deepgram.com/docs/flux/quickstart
17. Deepgram — Flux Multilingual GA press release (2026-04-29). https://deepgram.com/learn/deepgram-launches-flux-multilingual-press-release
18. Deepgram — Introducing Flux TTS (Aug 2026). https://deepgram.com/learn/introducing-flux-tts-conversation-native-text-to-speech-for-real-time-voice-agents
19. Deepgram — Flux TTS product page. https://deepgram.com/product/text-to-speech/flux
20. Deepgram — TTS streaming WebSocket reference. https://developers.deepgram.com/reference/text-to-speech/speak-streaming
21. Deepgram — TTS models (Aura-2 voices). https://developers.deepgram.com/docs/tts-models
22. Deepgram — Endpointing and interim results. https://developers.deepgram.com/docs/understand-endpointing-interim-results
23. Together AI — Deepgram Flux model page. https://www.together.ai/models/flux
24. AssemblyAI — Pricing. https://www.assemblyai.com/pricing
25. AssemblyAI — Universal-3 Pro Streaming (2026-03-03). https://www.assemblyai.com/blog/universal-3-pro-streaming
26. AssemblyAI — Streaming WebSocket API spec. https://www.assemblyai.com/docs/streaming/api-spec/streaming-websocket
27. AssemblyAI — Universal-3.5 Pro Realtime (2026-06-23; via search summary) and Artificial Analysis post. https://www.assemblyai.com/blog/universal-3-5-pro-realtime · https://x.com/ArtificialAnlys/status/2074160133702402314
28. ElevenLabs — Introducing Scribe v2 Realtime (2025-11-11). https://elevenlabs.io/blog/introducing-scribe-v2-realtime
29. ElevenLabs — API pricing. https://elevenlabs.io/pricing/api
30. ElevenLabs — Models. https://elevenlabs.io/docs/models
31. ElevenLabs — Realtime speech-to-text WebSocket API. https://elevenlabs.io/docs/api-reference/speech-to-text/v-1-speech-to-text-realtime
32. ElevenLabs — TTS stream-input WebSocket. https://elevenlabs.io/docs/api-reference/text-to-speech/v-1-text-to-speech-voice-id-stream-input
33. ElevenLabs — Multi-context WebSocket. https://elevenlabs.io/docs/api-reference/text-to-speech/v-1-text-to-speech-voice-id-multi-stream-input
34. OpenAI — API pricing. https://developers.openai.com/api/docs/pricing
35. OpenAI — gpt-live-transcribe model page. https://developers.openai.com/api/docs/models/gpt-live-transcribe
36. OpenAI — gpt-realtime-whisper model page. https://developers.openai.com/api/docs/models/gpt-realtime-whisper
37. OpenAI — Realtime transcription guide. https://developers.openai.com/api/docs/guides/realtime-transcription
38. OpenAI Cookbook — Migrate from Whisper to GPT-Transcribe and GPT-Live-Transcribe. https://developers.openai.com/cookbook/examples/migrating_from_whisper_to_gpt_transcribe
39. OpenAI Developer Community — GPT-Live-Transcribe and GPT-Transcribe (2026-07-29). https://community.openai.com/t/gpt-live-transcribe-and-gpt-transcribe-two-new-transcription-models-in-the-api/1388318
40. OpenAI — Text-to-speech guide. https://developers.openai.com/api/docs/guides/text-to-speech
41. OpenAI — gpt-4o-mini-tts model page. https://developers.openai.com/api/docs/models/gpt-4o-mini-tts
42. CostGoat — OpenAI Transcribe and Whisper API pricing (Sep 2026; deprecation dates reported, via search summary). https://costgoat.com/pricing/openai-transcription
43. Soniox — Models. https://soniox.com/docs/stt/models
44. Soniox — Pricing. https://soniox.com/pricing
45. Soniox — Real-time transcription API. https://soniox.com/docs/stt/rt/real-time-transcription
46. Speechmatics — Agent STT launch press release (2026-09-17). https://www.globenewswire.com/news-release/2026/09/17/3364138/0/en/speechmatics-launches-agent-stt-for-the-speech-errors-that-derail-voice-agents.html
47. Speechmatics — Realtime API reference. https://docs.speechmatics.com/rt-api-ref
48. Speechmatics — Pricing. https://www.speechmatics.com/pricing
49. Gladia — Pricing. https://www.gladia.io/pricing
50. Gladia — Live STT getting started. https://docs.gladia.io/chapters/live-stt/getting-started
51. Cartesia — Introducing Ink-2 (2026-07-09). https://www.cartesia.ai/blog/ink-2
52. Cartesia — Pricing (credits). https://docs.cartesia.ai/pricing
53. Cartesia — Sonic 3.6 model docs. https://docs.cartesia.ai/build-with-cartesia/tts-models/latest
54. Cartesia — TTS WebSocket API. https://docs.cartesia.ai/api-reference/tts/websocket
55. MarkTechPost — Cartesia ships Sonic-3.6 (2026-08-18). https://www.marktechpost.com/2026/08/18/cartesia-ships-sonic-3-6-a-streaming-tts-model-that-now-leads-both-artificial-analysis-speech-arenas/
56. Google — Gemini API pricing. https://ai.google.dev/gemini-api/docs/pricing
57. Google — Gemini API speech generation (TTS). https://ai.google.dev/gemini-api/docs/speech-generation
58. Google — Intelligent transcription with Gemini 3.5 Transcribe (2026-08-26). https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-3-5-transcribe/
59. Google Cloud — Chirp 3 transcription docs (via search summary). https://docs.cloud.google.com/speech-to-text/docs/models/chirp-3
60. Microsoft — MAI-Transcribe-2 (Tech Community, 2026-09-03; via search summary). https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/mai-transcribe-2-highest-quality-transcription-at-the-fastest-speed-and-lowest-c/4550972
61. Groq — Supported models and pricing. https://console.groq.com/docs/models
62. Cerebras — Inference models overview. https://inference-docs.cerebras.ai/models/overview
63. Mistral AI — Voxtral transcribes at the speed of sound (Voxtral Transcribe 2, 2026-02-04). https://mistral.ai/news/voxtral-transcribe-2/
64. Mistral AI — Speaking of Voxtral (Voxtral TTS, 2026-03-23). https://mistral.ai/news/voxtral-tts/
65. Meta — Meet Muse Voice Transcribe (2026-09-03). https://dev.meta.ai/resources/blog/meet-muse-voice-transcribe-streaming-speech-to-text/
66. xAI — Speech to Text docs. https://docs.x.ai/developers/model-capabilities/audio/speech-to-text.md
67. Hume AI — Pricing. https://www.hume.ai/pricing
68. Inworld — Realtime TTS-2. https://inworld.ai/realtime-tts-2
69. Rime — Launching Arcana v3 (2026-02-04). https://www.rime.ai/resources/arcana-v3
70. LMNT — homepage shutdown notice (checked 2026-09-24). https://www.lmnt.com/
71. Together AI — Speech-to-text docs. https://docs.together.ai/docs/speech-to-text
72. Anthropic — Models overview. https://platform.claude.com/docs/en/docs/about-claude/models/overview
73. Ollama — OpenAI compatibility. https://docs.ollama.com/api/openai-compatibility
74. llama.cpp — Server README. https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
75. LM Studio — OpenAI compatibility endpoints. https://lmstudio.ai/docs/developer/openai-compat
76. ml-explore — mlx-lm SERVER.md. https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/SERVER.md

**Local models and runtimes**
77. NVIDIA — Nemotron 3.5 ASR streaming 0.6B model card (2026-06-04). https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b
78. NVIDIA — NeMo-Speech.cpp. https://github.com/NVIDIA/NeMo-Speech.cpp
79. NVIDIA — Parakeet Realtime EOU 120M v1. https://huggingface.co/nvidia/parakeet_realtime_eou_120m-v1
80. NVIDIA — Parakeet TDT 0.6B v3. https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
81. Moonshine Voice (GitHub). https://github.com/moonshine-ai/moonshine
82. Useful Sensors — Moonshine Streaming model card. https://huggingface.co/UsefulSensors/moonshine-streaming-medium
83. Kyutai — STT model card. https://huggingface.co/kyutai/stt-1b-en_fr
84. Kyutai — delayed-streams-modeling (STT and TTS). https://github.com/kyutai-labs/delayed-streams-modeling
85. SYSTRAN — faster-whisper README and benchmarks. https://github.com/SYSTRAN/faster-whisper
86. ggml-org — whisper.cpp README. https://github.com/ggml-org/whisper.cpp
87. k2-fsa — sherpa-onnx README. https://github.com/k2-fsa/sherpa-onnx
88. k2-fsa — sherpa-onnx Kokoro models and RTF. https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/kokoro.html
89. istupakov — onnx-asr README and benchmarks. https://github.com/istupakov/onnx-asr
90. Qwen — Qwen3-ASR-1.7B model card. https://huggingface.co/Qwen/Qwen3-ASR-1.7B
91. Blaizzy — mlx-audio README. https://github.com/Blaizzy/mlx-audio
92. senstella — parakeet-mlx README. https://github.com/senstella/parakeet-mlx
93. Kyutai — Pocket TTS README and model card. https://github.com/kyutai-labs/pocket-tts · https://huggingface.co/kyutai/pocket-tts
94. hexgrad — Kokoro-82M model card. https://huggingface.co/hexgrad/Kokoro-82M
95. thewh1teagle — kokoro-onnx README. https://github.com/thewh1teagle/kokoro-onnx
96. OHF-Voice — piper1-gpl README. https://github.com/OHF-Voice/piper1-gpl
97. Resemble AI — Chatterbox-Turbo model card. https://huggingface.co/ResembleAI/chatterbox-turbo
98. Resemble AI — Chatterbox-Nano model card. https://huggingface.co/ResembleAI/chatterbox-nano
99. Resemble AI — Chatterbox-Flash model card. https://huggingface.co/ResembleAI/chatterbox-flash
100. Microsoft — VibeVoice-Realtime-0.5B model card. https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B
101. Qwen — Qwen3-TTS-12Hz-0.6B-Base model card. https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base
102. Fish Audio — S2 Pro model card. https://huggingface.co/fishaudio/s2-pro
103. ekwek — Soprano-1.1-80M model card. https://huggingface.co/ekwek/Soprano-1.1-80M
104. Neuphonic — NeuTTS Air model card. https://huggingface.co/neuphonic/neutts-air
105. nineninesix — KaniTTS 370M model card. https://huggingface.co/nineninesix/kani-tts-370m
106. NVIDIA — Magpie TTS Multilingual 357M model card. https://huggingface.co/nvidia/magpie_tts_multilingual_357m
107. Mistral AI — Voxtral-4B-TTS-2603 model card. https://huggingface.co/mistralai/Voxtral-4B-TTS-2603
108. BreezeBlue — Breeze-TTS-2 model card; license and latency via MindStudio and creativeaishow search summaries. https://huggingface.co/BreezeBlue/Breeze-TTS-2
109. Boson AI — higgs-tts-3-4b model card. https://huggingface.co/bosonai/higgs-tts-3-4b
110. Nari Labs — Dia2-2B model card. https://huggingface.co/nari-labs/Dia2-2B
111. Canopy Labs — Orpheus-TTS README. https://github.com/canopyai/Orpheus-TTS
112. FunAudioLLM — CosyVoice README (Fun-CosyVoice 3.0). https://github.com/FunAudioLLM/CosyVoice
113. index-tts — IndexTTS README (license). https://github.com/index-tts/index-tts
114. Hugging Face Hub metadata (licenses and sizes): https://huggingface.co/sesame/csm-1b · https://huggingface.co/maya-research/maya1 · https://huggingface.co/Zyphra/Zonos-v0.1-transformer · https://huggingface.co/SWivid/F5-TTS · https://huggingface.co/SparkAudio/Spark-TTS-0.5B · https://huggingface.co/myshell-ai/MeloTTS-English · https://huggingface.co/Supertone/supertonic-2
115. PyPI JSON API — versions, upload dates, wheel platforms and `requires_python` (queried 2026-09-24). https://pypi.org/pypi/{package}/json
116. GitHub Releases API — latest release tags (queried 2026-09-24). https://api.github.com/repos/{owner}/{repo}/releases/latest
117. Qwen — Qwen3.5-9B model card (BFCL-V4 and TAU2 table). https://huggingface.co/Qwen/Qwen3.5-9B
118. Google — Gemma 4 E4B-it model card (Tau2 table). https://huggingface.co/google/gemma-4-E4B-it
119. Technology.org — OpenAI GPT-6 Sol and Luna pricing and benchmarks (2026-09-23; via search summary, page returned 403). https://www.technology.org/2026/09/23/openai-gpt-6-sol-luna-pricing-benchmarks/
120. OrcaRouter — Inworld Realtime TTS-2 GA and Flash (via search summary). https://www.orcarouter.ai/blog/inworld-realtime-tts-2-flash-launch
121. Smallest.ai — Text to speech (Lightning v3.1). https://smallest.ai/text-to-speech
