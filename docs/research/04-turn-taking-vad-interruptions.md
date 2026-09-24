# 04 — Turn-taking, VAD, interruptions, full-duplex and audio processing

*Research date: 2026-09-24. Method: vendor docs, model cards, papers; source reading of Pipecat (9d4c508), LiveKit Agents (bf92dd0) and LiveKit Python SDK (ee527bd), all 2026-09-23; PyPI wheel metadata (2026-09-24). Benchmark numbers are **vendor-reported** unless stated. Unconfirmed items are marked (unverified).*

## 1. Executive summary

- **Default VAD: Silero v6.2.x (MIT).** 32 ms windows, 8/16 kHz only, <1 ms per chunk on one CPU thread; v6 cut errors on noisy real-life audio by 16% vs v5; latest v6.2.3 (2026-09-23) [1][2]. Ship the ONNX file itself: `pip install silero-vad` pulls in torch [3].
- **Semantic end-of-turn (EOT) on top of VAD is now standard.** Pipecat defaults to Smart Turn v3 (audio-only, 8 MB int8 ONNX, BSD-2, 23 languages) [4][5]. LiveKit shipped the audio-native Turn Detector v1/v1-mini on 2026-06-17 [6]. Deepgram Flux, AssemblyAI and Kyutai/Gradium build EOT into STT [7][8][9][10].
- **Detectors differ a lot, but benchmarks are vendor-run.** LiveKit's eot-bench (English) false-cutoff rates at a 300 ms latency budget: LiveKit v1 9.9%, Deepgram Flux 12.9%, LiveKit v1-mini 27.8%, Smart Turn v3.2 35.2%, AssemblyAI 49.4%, silence-only VAD 55.6% [11]. Krisp's benchmark ranks Krisp first [12]. We must reproduce, not cite.
- **Licenses rule out most detectors as defaults.** LiveKit's models may only be used "together with the LiveKit Agents framework" [13]. TEN VAD and TEN Turn Detection ban competing with Agora [14][15]. Vogent-Turn-80M may not be a voice-agent platform's default [16]. Among production-ready options, Smart Turn (BSD-2) is the only permissive, local, multilingual audio EOT model; NAMO (Apache-2.0) is text-only [17].
- **The endpointing policy matters as much as the model.** It has three knobs: threshold, minimum silence before acting, maximum hold [11]. Defaults: LiveKit 0.3/2.5 s with its audio detector [18]; Pipecat VAD stop 0.2 s plus a 3 s Smart Turn fallback [19]; OpenAI `server_vad` 500 ms [20].
- **Frameworks start the LLM before the turn is confirmed.** LiveKit does it by default (LLM, not TTS) [21]. Deepgram's EagerEndOfTurn comes 150–250 ms early for 50–70% more LLM calls [7][22]. Pipecat holds the early output until confirmation [5].
- **Barge-in needs more than VAD.** Krisp measured VAD-only barge-in at 66.3% false positives. Krisp's interruption model got 5.9%. A min-word rule got 3.6% but took 1.53 s on average to fire, vs 0.83 s for the model [12]. LiveKit's cloud model rejects 51% of VAD barge-ins [23]. We found no permissive open-weights interruption classifier.
- **On a possible false barge-in, pause rather than cancel.** LiveKit pauses playback and resumes if no transcript arrives within 2.0 s of the user going quiet [21]. When the barge-in is real, cut the assistant turn back to what was actually heard (OpenAI `conversation.item.truncate` with `audio_end_ms`) [24].
- **Local echo cancellation is the biggest cross-platform risk.** Best Python route: the WebRTC Audio Processing Module (APM) in the `livekit` wheel (Apache-2.0; Linux x86_64/aarch64, macOS x86_64/arm64, Windows x64). LiveKit's console mode does exactly this [25][26][27]. OS-level processing is inconsistent [28][29].
- **Packaging trap:** onnxruntime ≥1.24 (Feb 2026) needs Python ≥3.11, has no Intel-Mac wheels, and needs macOS 14+ on Apple Silicon [30].
- **Full-duplex speech-to-speech models are not ready as defaults.** Moshi takes the floor in 98.5% of test pauses [31]. In the ICASSP 2026 HumDial challenge, pipelines with dedicated decision modules ranked top. Moshi (35.4) and Freeze-Omni (29.6) were near the bottom on interruption handling [32].

## 2. Voice activity detection (VAD)

### 2.1 Models

| VAD | Type / size | Audio | Speed | License | Python packaging |
|---|---|---|---|---|---|
| **Silero v6.2.3** | small neural net, ~2 MB; wheel ships several ONNX builds (op15, op18 "ifless", half, OpenVINO, 16 kHz sequence) | 8/16 kHz; 512 samples (32 ms) at 16 kHz, 256 at 8 kHz | v5: 189 µs per 31.25 ms chunk, ONNX, 1 thread | MIT | `silero-vad` requires torch; ONNX Runtime optional [2][33][34][3] |
| **TEN VAD** | native lib (306 KB on Linux) + ONNX | 16 kHz only; 10 or 16 ms hops | real-time factor 0.009–0.016 | Apache-2.0 + non-compete | `ten-vad` bundles Linux x64, Windows and macOS libs; **no Linux aarch64** [35][14][36] |
| **WebRTC VAD** | classic, non-neural | 8/16/32/48 kHz; 10/20/30 ms frames; modes 0–3 | negligible | BSD-style / MIT | use `webrtcvad-wheels` 2.0.14 (all OSes) [37][38] |
| **Picovoice Cobra v3** | proprietary | set by SDK | v3.0.0 (Dec 2025) added GPU/multi-core | SDK Apache-2.0; engine needs a Picovoice AccessKey | `pvcobra` 3.0.3 [39][40] |
| **NVIDIA MarbleNet v2.0** | CNN, 91.5K params | 16 kHz; one probability per 20 ms; 6 languages | n/a | NVIDIA Open Model License | NeMo; ONNX export [41] |
| **Energy gate** | RMS / loudness | any | trivial | — | Pipecat requires Silero confidence **and** loudness ≥0.6 [42] |

Commercial VADs (ai-coustics Quail, Krisp VIVA) plug into Pipecat [43].

**Accuracy (both sides vendor-run).** Silero's wiki reports multi-domain ROC-AUC of v6 0.97, v5 0.96, TEN 0.93, WebRTC 0.73. On noise-only audio (ESC-50), accuracy is v6 0.87 vs TEN 0.42 [44]. TEN claims the best precision-recall on its own test set and says Silero lags "several hundred milliseconds" at speech offsets [35]. Silero lists voice-like music and very high-pitched voices as known v6 weak spots; v6.2 improved child voices and "lower quality phone calls" [1]. **Implication:** our benchmark must measure both per-frame accuracy and how late the offset is detected.

### 2.2 Parameters in practice

| Setting | Silero lib | LiveKit Silero plugin | Pipecat `VADParams` | OpenAI `server_vad` | Gemini Live |
|---|---|---|---|---|---|
| speech threshold | 0.5 | 0.5 | 0.7 (+ loudness ≥0.6) | 0.5 | sensitivity enum |
| silence threshold | threshold − 0.15 | max(threshold − 0.15, 0.01) | — | — | sensitivity enum |
| min speech | 250 ms (offline) | 0.05 s | `start_secs` 0.2 | — | `prefixPaddingMs`* |
| min silence | 100 ms | 0.55 s | `stop_secs` 0.2 | 500 ms | `silenceDurationMs` |
| audio kept before speech | 30 ms each side | 0.5 s | (Smart Turn keeps 0.5 s) | 300 ms | — |

Sources: [33][45][42][19][20][46].

\*Trap: Gemini's `prefixPaddingMs` is "the required duration of detected speech before start-of-speech is committed". OpenAI's `prefix_padding_ms` is how much audio to keep *before* detected speech [46][20].

**How the frameworks wire VAD.**
- **Pipecat:** VAD states QUIET → STARTING → SPEAKING → STOPPING, confirmed by frame counts [42]. A user turn starts on VAD or a transcript; it stops when Smart Turn v3 says so [5].
- **LiveKit:** the Silero plugin sends padded speech-start/speech-end events to the turn logic. It pins ONNX Runtime to 1 thread and turns off spin-waiting to avoid wasting CPU [45]. The audio turn detector needs VAD silence of at least 0.25 s [18].

## 3. Semantic end-of-turn detection

### 3.1 Landscape

| Detector | Input | Size / cost | Languages | License / availability |
|---|---|---|---|---|
| **Smart Turn v3.2** (v3.0 2025-09-11; v3.1 2025-12-03; v3.2 2026-01-07) | audio only, up to 8 s at 16 kHz; needs a VAD | Whisper-Tiny encoder + linear head, 8M params; 8 MB int8 (CPU) or 32 MB (GPU) ONNX; 12.6 ms on AWS c7a.2xlarge, 94.8 ms on t3.medium (v3.0 figures) | 23 | BSD-2 (weights, data, training code) [47][48][49][4][50] |
| **LiveKit Turn Detector v1 / v1-mini** (2026-06-17) | audio (fuses a semantic branch with an acoustic branch) | v1: LiveKit's cloud. v1-mini: local CPU, runs as a native library; size not published | 14 | v1: free on LiveKit Cloud. v1-mini: LiveKit Model License, **only usable with LiveKit Agents** [6][18][13][21] |
| LiveKit text detector (deprecated; removal in Agents 2.0) | transcript + context | Qwen2.5-0.5B-based, 396 MB, ~50–160 ms per turn | 14 | LiveKit Model License [18][51] |
| **TEN Turn Detection** | text | Qwen2.5-7B-based; outputs finished / wait / unfinished | English, Chinese | Apache-2.0 + bans on end-user devices and on competing with Agora [52][15] |
| **Vogent-Turn-80M** (Oct 2025) | audio + previous and current text | Whisper-Tiny + 12-layer SmolLM2 (~80M params); ~7 ms on a T4 GPU; CPU not yet optimized | English | Code Apache-2.0; weights: platforms may not make it their default [53][16][54] |
| **NAMO v1** | text | mmBERT-base, ~295 MB; <29 ms quantized | 23 | Apache-2.0 [17] |
| **UltraVAD** | audio + dialog history | Llama-8B backbone; 65–110 ms on an A6000 GPU | 26 | open weights, license not stated (unverified) [55][56] |
| **Krisp Turn Prediction v3 / Interruption Prediction v1** (May 2026) | audio | ~9M params / 30 MB, and ~6M / 24 MB; CPU | 12+ / English | proprietary SDK; Pipecat integration [12][57] |
| **Deepgram Flux** | STT with built-in EOT | defaults: `eot_threshold` 0.7, `eot_timeout_ms` 5000; optional `eager_eot_threshold` 0.3–0.9 | English + multilingual (10) | commercial API [58][59] |
| **AssemblyAI Universal-Streaming** | STT with built-in EOT (semantic, with silence fallback) | defaults: threshold 0.4, min silence 400 ms, max silence 1280 ms | multilingual | commercial API [8] |
| **OpenAI `semantic_vad`** | server classifier on the user's words | eagerness low / medium / high wait at most 8 / 4 / 2 s (auto = medium) | — | API [60][20] |
| **Gemini Live automatic activity detection** | server-side VAD | start/end sensitivity settings plus `prefixPaddingMs` / `silenceDurationMs`; defaults not documented | — | API [46][61] |
| **Kyutai STT / Gradium** | STT that also predicts pauses of 0.5 / 1 / 2 / 3 s | Kyutai: Rust server only; Gradium: forecast every 80 ms | Kyutai 1B: EN/FR | Kyutai license not checked; Gradium is an API [9][10] |

### 3.2 Benchmarks (read critically)

**eot-bench** is LiveKit's benchmark (code Apache-2.0, data CC-BY-4.0) [11][62]:
- **Data:** real human-to-agent turns in 14 languages. Every pause of at least 100 ms is labelled *hold* (the user continues) or *eot* (the turn ended).
- **Method:** it replays audio causally. It then sweeps the decision threshold, the minimum silence before acting (`action_delay`) and the maximum hold (timeout).

English results:

| Model | False cutoffs @300 ms | @600 ms | Latency @5% cutoffs | @10% |
|---|---:|---:|---:|---:|
| LiveKit v1 | 9.9% | 4.5% | 543 ms | 295 ms |
| JoinIn AI Baton | 12.3% | 4.8% | 577 ms | 350 ms |
| Deepgram Flux | 12.9% | 9.9% | 1151 ms | 548 ms |
| ultraVAD | 27.7% | 11.9% | 899 ms | 663 ms |
| LiveKit v1-mini | 27.8% | 12.1% | 1070 ms | 698 ms |
| Smart Turn v3.2 | 35.2% | 14.8% | 1051 ms | 739 ms |
| VAP (silent agent) | 46.9% | 14.6% | 1131 ms | 749 ms |
| AssemblyAI | 49.4% | 14.6% | 1049 ms | 713 ms |
| Silence-only VAD | 55.6% | 21.7% | 1600 ms | 1000 ms |

Other vendors' numbers:
- **Krisp:** Turn Prediction v3 balanced accuracy 88.05, Deepgram Flux 87.10, LiveKit 82.70 (version unstated), Smart Turn v3.2 77.41 [12].
- **UltraVAD:** beats Smart Turn v2 on turns that need dialog context (77.5% vs 63.0%) but ties on single-turn tests (93.7% vs 94.3%) [55].
- **Smart Turn v3.1:** English 94.7% (95.6% for the GPU model) [48].

**Takeaways:**
- Every learned model beats silence alone.
- Smart Turn is far behind the best proprietary models at a 300 ms budget, and about 10 points behind at 600 ms.
- Smart Turn can be fine-tuned per language. TamilEOT raised Tamil accuracy from 70.30% zero-shot to 83.71% [63].

### 3.3 Research to track

- **VAP:** predicts both speakers' near-future voice activity from stereo audio; runs in real time on CPU; multilingual [64].
- **TurnGPT:** a text language model that uses dialog context [65].
- **Easy Turn:** classifies complete / incomplete / backchannel / wait from acoustics plus text; 1,145 h of training data [66].
- **Next-Turn (2026):** predicts the time until the next speech onset; +25.9 points of endpoint accuracy within 320 ms [67].
- **SID-Bench (2026):** benchmark and "Average Penalty Time" metric for semantic interruption detection [68].
- **Talking Turns (ICLR 2025):** audio foundation models "interrupt too aggressively" and "rarely backchannel" [69].

## 4. Endpointing strategies

1. **Silence timeout only.** OpenAI waits 500 ms; Pipecat's speech timeout waits 0.6 s, then for the final transcript [20][5]. This is the worst option on eot-bench [11].
2. **Model plus min/max delay.** LiveKit waits `min_delay` after VAD offset. If P(EOT) is below the "unlikely" threshold, it waits `max_delay` instead [21]. Pipecat ends an "incomplete" turn after 3 s anyway [19].
3. **Dynamic endpointing (LiveKit).** `min_delay` is learned per session as a moving average (α=0.9). It learns from pauses inside a user turn, and from pauses where the user cut the agent off right after it started (a sign it answered too early). It is clamped between min and max, and backchannels are ignored [21].
4. **Starting the reply early.**
   - LiveKit: on by default for the LLM only; skipped after 10 s of user speech; at most 3 retries per turn [21].
   - Deepgram Flux: EagerEndOfTurn starts the LLM, TurnResumed cancels it; the final transcript "will exactly match" the early one [70][58].
   - Pipecat: holds the early reply until the turn is confirmed and withdraws it after 5 s [5].
5. **Let the LLM decide.** Pipecat's `FilterIncompleteUserTurnStrategies` makes every LLM reply start with a marker: ● complete, ◐ cut off (re-prompt after 5 s), ○ needs more time (10 s) [5].
6. **Finalizing the transcript fast.** After an EOT, Kyutai pushes audio faster than real time so its 0.5 s-delay STT finishes in ~125 ms (their "flush trick") [9]. Pipecat stops waiting as soon as the STT marks the transcript final [5].
7. **Pauses inside numbers, emails and addresses.** Phone numbers read in groups trip silence detectors [71]. Fixes:
   - Longer silences while the user dictates. AssemblyAI's conservative preset: 0.7 threshold, 800/3600 ms [8].
   - Changing settings mid-session: AssemblyAI `UpdateConfiguration`/`ForceEndpoint` [71]; Deepgram Configure messages (unverified).
   - Per-agent overrides (LiveKit) [21]; OpenAI's `low` eagerness, which waits up to 8 s [20].

   For reference, humans switch turns in ~208 ms on average (Stivers et al. 2009, as cited by Gradium) [10].

## 5. Interruptions and barge-in

### 5.1 Telling real barge-ins from backchannels and noise

Krisp's comparison (vendor-run) [12]:

| Method | Mean time to interrupt | False-positive rate |
|---|---:|---:|
| VAD only | 0.375 s | 66.3% |
| Minimum word count | 1.528 s | 3.6% |
| Krisp Interruption Prediction v1 @0.4 | 0.833 s | 5.9% |

**LiveKit adaptive interruption** (Mar 2026) is an audio model (encoder + CNN) that runs only in LiveKit Cloud [23][72][21]:
- ≤30 ms inference; 86% precision / 100% recall at 500 ms of overlapping speech; needs 216 ms of audio (median).
- The client sends ≥50 ms of audio plus a 1.0 s prefix, in windows of up to 3 s.
- It ignores backchannels ("uh-huh", "okay") in the first and last 1.0 s of an agent turn.

**Framework defaults:**
- LiveKit: minimum interruption 0.5 s of speech, 0 words [21].
- Pipecat: applies its minimum word count only while the bot speaks [5].
- Gemini: interrupts on user speech by default, and the cut-off reply is discarded; `NO_INTERRUPTION` turns this off [46][61].

### 5.2 Recovering from a false interruption (LiveKit)

1. When overlapping speech qualifies, *pause* agent audio (if the output supports pausing).
2. When the user stops, start `false_interruption_timeout` (2.0 s).
3. If an end-of-turn decision is pending, wait for it.
4. If no transcript or committed turn appeared, resume and emit `agent_false_interruption(resumed=True)`; otherwise interrupt for real.

While the agent is uninterruptible, user audio is thrown away by default (`discard_audio_if_uninterruptible=True`) [21][73].

### 5.3 Trimming the context to what was actually heard

- **OpenAI Realtime:** with WebRTC/SIP the server tracks playback and trims automatically. With WebSocket the client must stop playback and send `conversation.item.truncate` with `audio_end_ms`. OpenAI says transcript alignment is only approximate [24][20].
- **Pipecat** truncates at min(time since playback started, length of generated audio) [74].
- **LiveKit** writes only the text that was actually played to the chat context, marked as interrupted. For realtime models it truncates at the playback position [21].
- **Why it matters:** IHBench (2026) checks whether an agent resumes at the right step without repeating what the user heard; open-weight models degrade faster [75]. Practitioners give the same advice [76].

### 5.4 Echo-induced self-interruption

If the agent's own audio leaks into the mic, VAD fires and the agent interrupts itself [76]. Fixes:
- echo cancellation with the right reference signal and delay (§6);
- VAD plus a minimum duration plus semantic checks before interrupting;
- checking detected speech against playback timing [76];
- where OS echo cancellation is weak, gating the mic for 500–800 ms after TTS ends, disarmed as soon as new TTS starts (iOS field report) [29].

## 6. Echo cancellation and audio processing

### 6.1 Python options

| Package | What it does | Wheels (Linux / macOS / Windows) | License | Last release |
|---|---|---|---|---|
| **`livekit`** (`rtc.AudioProcessingModule`) | WebRTC APM: echo cancellation, noise suppression, high-pass filter, gain control. 10 ms frames; takes the played audio as reference; accepts a delay hint | x86_64+aarch64 / x86_64+arm64 / x64 | Apache-2.0 | 1.1.20, 2026-09-23 [25][27] |
| `aec-audio-processing` | WebRTC AudioProcessing 2.x | Windows x64 wheel only; elsewhere build from source | BSD-3 | 2025-09 [77] |
| `webrtc-audio-processing` | old APM wrapper | Linux armv7 only | BSD | 2019, unmaintained [78][79] |
| `pyaec` | SpeexDSP echo cancellation (via aec-rs) | all three OSes | MIT | 2024-12 [80][81] |
| `speexdsp` / `speexdsp-ns` | Speex | source only / Linux, noise suppression only | BSD | unmaintained [82][83] |
| **`pyrnnoise`** | RNNoise v0.2 noise suppression; 48 kHz, 10 ms frames; also gives a speech probability | all three OSes, but the macOS wheel needs macOS 26+ (0.4.3: 15+); no source package | Apache-2.0 wrapper, BSD-3 RNNoise | 2026-09-23 [84][85][86] |
| DeepFilterNet | 48 kHz full-band noise suppression | `deepfilterlib` on all three OSes; the Python API needs torch | MIT/Apache-2.0 | 2023-08 [87][88] |
| Krisp (NC, BVC, VIVA), ai-coustics, NVIDIA Maxine | commercial noise suppression, background-voice cancellation and voice isolation (Maxine needs an NVIDIA GPU; Windows/Linux) | vendor SDKs | proprietary | [89][43][90] |

**Reference implementation: LiveKit console mode** [26]:
- uses `sounddevice` at 24 kHz;
- runs every 10 ms mic frame through the APM (all four features on);
- passes every played frame to `process_reverse_stream`;
- sets the stream delay from PortAudio's timestamps.

WebRTC AudioProcessing 2.0 (Jan 2025) moved to WebRTC M131 and improved echo cancellation [91]. Current libwebrtc's echo canceller is AEC3; we did not verify which canceller the `livekit` build enables (unverified).

### 6.2 OS-level voice processing

- **macOS:** `AVAudioIONode.setVoiceProcessingEnabled` (macOS 10.15+) and the VoiceProcessingIO audio unit [92][93].
  - Python can reach them via PyObjC (`pyobjc-framework-AVFoundation`), but using this for a full duplex audio path is unverified [94].
  - Pitfalls from the field: enable it after attaching the playback graph; use the voiceChat mode; manual rendering is not supported [29].
- **Windows:** apps tag streams (e.g. Communications) and the hardware vendor decides the effects, so "not all modes might be available" [95].
  - Windows 11 (build 22540+) lets apps pick the output device used as the echo reference [96].
  - Vendor echo-cancellation plug-ins that use private driver channels often ignore USB and Bluetooth devices [28].
  - It is unverified whether PortAudio exposes stream categories. PyAudioWPatch can capture system playback (WASAPI loopback) to use as an echo reference [97].
- **Linux:** PipeWire's `module-echo-cancel` creates echo-cancelled virtual devices using WebRTC [98]. PulseAudio has a similar module (unverified). Both need user session config, so document them rather than enable them by default.

**Guidance:**
- Don't run noise suppression twice; LiveKit warns it "can also interfere with turn detection and reduce transcription quality" [89].
- Keep one resampling stage: APM wants 10 ms frames, Silero 8/16 kHz, Smart Turn 16 kHz, RNNoise/DeepFilterNet 48 kHz.

## 7. Full-duplex dialogue

"Full-duplex" models listen and speak at the same time, instead of taking strict turns.

| Model | Approach | Notes | License |
|---|---|---|---|
| dGSLM (2022) | two transformers with cross-attention, trained on 2,000 h of two-channel phone audio, no text | natural overlaps and laughter | research [99] |
| Moshi (2024) | Helium LM + Mimi codec (12.5 Hz); separate user and agent streams | 160 ms theoretical latency, ~200 ms on an L4 GPU; PyTorch build needs a 24 GB GPU (MLX build for Macs) | weights CC-BY 4.0 [100][101] |
| SyncLLM (2024) | Llama-3-8B kept in sync with the wall clock | tolerates 240 ms network latency | research [102] |
| Freeze-Omni (2024) | frozen LLM + per-chunk speak/listen prediction | better pause handling | [103] |
| OmniFlatten (2024) | speech and text flattened into one stream; trained half- then full-duplex | — | [104] |
| SALMONN-omni (2025) | no audio codec; learned tokens decide when to speak or listen | claims ≥30% relative gain; handles echo and barge-in | [105] |
| PersonaPlex (NVIDIA, Jan 2026) | 7B, built on Moshi, role and voice prompts | reports better latency, naturalness and role adherence than the systems it compares against | weights: NVIDIA Open Model License [106][107][108] |
| SoulX-Duplug (Mar 2026) | plug-in module for pipelines: streaming ASR + dialog-state prediction | open-sourced | [109] |

**Findings.**
- **Full-Duplex-Bench v1** [31]:
  - How often each model takes the floor during a user's pause (lower is better): Moshi 0.985, dGSLM 0.934, Freeze-Omni 0.642, Gemini Live 0.255.
  - Response latency on smooth turn changes: Moshi 0.265 s, Gemini Live 1.301 s.
  - After interruptions, Moshi answers fast but off-topic: relevance score 0.765 vs Freeze-Omni 3.615 (GPT-4o judge).
  - Later versions: v1.5 adds overlap scenarios, v2 adds real-time evaluation, v3 adds disfluency and tool use [110].
- **HumDial (ICASSP 2026)** scores handling of real interruptions (Int) and ignoring backchannels, pauses and third-party speech (Rej) [32]:
  - Winners: Cookie ASR (Int 79.3 / Rej 72.2) and Badcat (89.7 / 57.8).
  - Gemini-2.5: 79.8 / 36.5.
  - Freeze-Omni 29.6 / 50.2, Moshi 35.4 / 22.8.
  - Teams found that specialized modules filtering noise before the LLM made systems more robust.
- **Surveys:** pipeline systems with an external decision layer "remain competitive" on latency. Training data and the latency-vs-coherence trade-off remain open [111][112].
- **Backchannel generation** (the agent saying "mm-hmm"):
  - Audio models rarely do it [69][31].
  - LiveKit has an internal "backchannel opportunity" event, and its v1 detector outputs a backchannel probability [21].

## 8. Implications for voice-agent-next

### 8.1 Architecture

- **One owner.** A single frame-clocked `TurnController` owns the floor state.
- **Plugins.** Everything else is a pluggable Protocol:
  - `Vad`;
  - `EotDetector` (audio, text, STT-built-in, or realtime-model);
  - `InterruptionClassifier`, `EchoCanceller`, `NoiseSuppressor`;
  - `PlaybackSink`, which **must** report the playback position and support pause/resume.
- **Realtime backends** (OpenAI, Gemini, Moshi) map their server events onto the same states. voice-agent-next still owns truncation and false-interruption handling.

### 8.2 Turn-taking state machine

**States.**
- `IDLE`
- `USER_SPEAKING`
- `USER_PAUSED` — waiting on the EOT decision; an early LLM call is allowed.
- `AGENT_PREPARING` — turn committed, nothing audible yet.
- `AGENT_SPEAKING`
- `OVERLAP` — user speaks during agent speech; agent audio is paused.
- `AGENT_PAUSED` — waiting to decide whether the interruption was false.

**Events.**
- VAD and STT: `vad_start`, `vad_end`, `eot(p)`, `stt_partial`, `stt_final`, `stt_eot`.
- Timers: `t_min`, `t_max`, `t_fi`.
- Playback: `playout_started`, `playout_progress(pos)`, `playout_done`.
- Interruption verdict: `interrupt`, `backchannel`, `noise` or `echo`.
- Control: `force_commit` (from the app, a tool or push-to-talk), `mute`.

| From | Event / condition | To | Actions |
|---|---|---|---|
| IDLE | `vad_start` | USER_SPEAKING | open the turn with the pre-speech audio |
| USER_SPEAKING | `vad_end` (silence ≥ `candidate_pause`) | USER_PAUSED | run EOT on the last ≤8 s. If p ≥ θ: arm `t_min` and start the LLM early. Otherwise arm `t_max` |
| USER_PAUSED | `vad_start` | USER_SPEAKING | cancel timers; withdraw the early reply |
| USER_PAUSED | `t_min` / `t_max` / `stt_eot` / `force_commit` | AGENT_PREPARING | commit the turn; wait ≤ `stt_final_timeout` for the final transcript; keep the early reply only if the transcripts match |
| AGENT_PREPARING | `vad_start` | USER_SPEAKING | cancel generation; merge the new speech into the same turn; update dynamic endpointing |
| AGENT_PREPARING | `playout_started` | AGENT_SPEAKING | start tracking the playback position |
| AGENT_SPEAKING | `playout_done` | IDLE | commit the full assistant message |
| AGENT_SPEAKING | `vad_start`, interruptions allowed | OVERLAP | pause playback; classify (§8.3) |
| AGENT_SPEAKING | `vad_start`, not interruptible | AGENT_SPEAKING | drop the user audio |
| OVERLAP | `interrupt` | USER_SPEAKING | stop, cancel, truncate; the user turn starts at the overlap start |
| OVERLAP | `vad_end` with no verdict | AGENT_PAUSED | arm `t_fi` |
| OVERLAP / AGENT_PAUSED | `backchannel` / `noise` / `echo` | AGENT_SPEAKING | resume |
| AGENT_PAUSED | `t_fi` fires with no meaningful words | AGENT_SPEAKING | resume; emit `false_interruption` |

**Default parameters.** "Proposal" means our own starting value, to be tuned by the benchmark suite.

| Parameter | Default | Basis |
|---|---|---|
| analysis sample rate / echo-cancel frame | 16 kHz mono / 10 ms | [33][25] |
| VAD speech / silence threshold | 0.5 / 0.35 | Silero and LiveKit [33][45] |
| speech needed to start a turn | 0.10 s | proposal (LiveKit 0.05 s, Pipecat 0.2 s) |
| audio kept before speech | 0.5 s | [45][19] |
| `candidate_pause` (VAD silence) | 0.25 s | LiveKit minimum; Pipecat 0.2 s [18][19] |
| EOT threshold θ | 0.5 | Smart Turn [113] |
| `min_delay` / `max_delay` | 0.4 s / 2.5 s | proposal (LiveKit: 0.3/2.5 audio, 0.5/3.0 legacy) [18] |
| VAD-only silence timeout | 0.6 s | Pipecat 0.6 s, OpenAI 0.5 s [5][20] |
| dictation mode min / max delay | 1.0 s / 5.0 s | proposal (AssemblyAI conservative 0.8/3.6 s) [8] |
| dynamic endpointing | off (α 0.9 when on) | [21] |
| early LLM call | on for LLM, off for TTS; skip if user spoke >10 s; ≤3 retries | [21] |
| `stt_final_timeout` | 0.5 s | proposal; tune per STT provider |
| minimum interruption | 0.5 s (0.25 s once echo cancel is verified and a classifier is set) | LiveKit; proposal [21] |
| minimum interruption words | 0 (1 real word when streaming STT is available) | [21][12] |
| false-interruption timeout / resume | 2.0 s / on | [21] |
| backchannel window at start / end of agent turn | 1.0 s / 1.0 s (with a classifier only) | [21] |
| mic gate after playback, when there is no echo cancellation | 0.3 s | proposal (iOS report used 0.5–0.8 s) [29] |

### 8.3 Interruption handling algorithm

The pseudo-code below covers all three parts: deciding on an interruption, recovering from a false one, and truncating the context.

```python
# Illustrative pseudo-code only — not a library API.
on vad_start while state == AGENT_SPEAKING:
    if not allow_interruptions:
        drop_audio(); return                        # discard_audio_if_uninterruptible
    overlap_t0 = now(); playback.pause(); state = OVERLAP

on frame while state in (OVERLAP, AGENT_PAUSED):
    dur   = now() - overlap_t0
    words = meaningful_words(stt_partial)           # strip localized backchannel lexicon
    if echo_suspected(words, recent_tts_text, aec_ok): # proposal: text similarity if AEC off/unknown
        resume(); return
    if (clf and clf.p_interrupt >= clf.threshold) or \
       (dur >= min_interruption_duration and (min_words == 0 or len(words) >= min_words)):
        confirm_interruption()

on vad_end while state == OVERLAP:
    state = AGENT_PAUSED; start_timer("fi", false_interruption_timeout)   # 2.0 s

on timer("fi"):
    if eot_decision_pending(): defer_until_settled()
    elif not meaningful_words(stt_final_or_partial):
        resume(); emit("false_interruption", resumed=True)
    else:
        confirm_interruption()

def confirm_interruption():
    played_ms = playback.samples_rendered_ms() - playback.output_latency_ms()
    playback.stop(); tts.cancel(); llm.cancel()
    spoken = words_with_end_before(played_ms, tts_word_timestamps) or estimate_by_chars(played_ms)
    context.add_assistant(spoken, interrupted=True, unspoken_tail=...)
    if realtime_backend:
        realtime_backend.truncate(item_id, audio_end_ms=min(played_ms, generated_ms))
    state = USER_SPEAKING  # user turn starts at overlap_t0 (+ prefix padding)
```

Key design choices:
- `played_ms` subtracts the device's output latency, as LiveKit's console does [26].
- The truncation length follows Pipecat [74].
- The unspoken tail is kept as metadata, so the agent can resume without repeating itself [75].

### 8.4 Default local components

| Role | Default (license) | Optional adapters (caveat) |
|---|---|---|
| VAD | Silero v6.2 ONNX file bundled with our package, ONNX Runtime on 1 thread with no spin-waiting (MIT) [45]; fallback `webrtcvad-wheels` (MIT) | TEN VAD (non-compete), Cobra (AccessKey), MarbleNet (NVIDIA license), Krisp/ai-coustics (commercial) |
| End of turn | Smart Turn v3.2, CPU int8 ONNX (BSD-2) | NAMO (Apache-2.0, text); STT-built-in EOT (Flux, AssemblyAI, Soniox, Gradium); realtime server VAD; UltraVAD (GPU, license unclear); Vogent (can't be default); LiveKit models (**don't bundle**); Krisp (commercial) |
| Interruption | rules (duration + words + backchannel list) + false-interruption recovery | Krisp Interruption Prediction; LiveKit adaptive (LiveKit Cloud only) |
| Echo cancel / noise suppression / gain | WebRTC APM from the `livekit` wheel (Apache-2.0), on when playing through speakers | `pyaec` (MIT); PipeWire echo-cancel; macOS voice processing via PyObjC; Windows Communications mode |
| Extra noise suppression | none (APM's is enough) | RNNoise via `pyrnnoise`, DeepFilterNet, commercial SDKs |
| Audio I/O | `sounddevice` (MIT) [114] | WebRTC or telephony transports (client-side echo cancel) |

### 8.5 Benchmark suite hooks

- **End of turn:** vendor eot-bench (Apache-2.0 code, CC-BY data). Report false cutoffs @300/600 ms and latency @5/10% per language, always next to the silence-only VAD baseline. Use the sweep to set each detector's `min_delay`/`max_delay` [11][62].
- **Interruptions:** record overlap sets (backchannels, coughs and noise, real barge-ins, echo only). Measure false-positive rate, mean time to interrupt, median audio needed, and how often playback resumes after a false interruption [12][23].
- **Recovery:** IHBench-style checks: resumes at the right step, no repeated content [75].
- **Echo:** self-interruption rate with loudspeaker playback per OS and echo-cancel backend.
- **Full-duplex backends:** Full-Duplex-Bench v1 metrics plus the v2 real-time harness [31][110].

### 8.6 Pitfalls

1. **Model licenses are not code licenses.** Examples: LiveKit Model License, TEN non-compete, Vogent's default clause, UltraVAD has no stated license [13][14][16].
2. **onnxruntime ≥1.24:** Python ≥3.11, no Intel-Mac wheels, macOS 14+ on Apple Silicon. Pin `<1.24` or drop Intel Macs [30].
3. **`silero-vad` from pip pulls in torch** — bundle the ONNX file [3].
4. **ONNX Runtime thread pools spin and waste CPU** — use 1 thread, no spinning [45].
5. **Rigid input formats:**
   - Silero accepts only fixed windows at 8 or 16 kHz.
   - Smart Turn takes at most 8 s, padded at the *start*.
   - APM needs exactly 10 ms frames and a correct delay hint [33][4][25].
6. **Same name, different meaning:** OpenAI's and Gemini's "prefix padding" [46][20].
7. **Smart Turn needs a short VAD stop time (0.2–0.25 s).** A 0.5–0.8 s stop time adds that much latency to every reply [19][18].
8. **Audio-only EOT ignores dialog context** (e.g. a bare "yes" as an answer). Offer text or context-aware detectors [55].
9. **Early LLM calls cost 50–70% more.** Never let an early reply reach the context or the speaker before the turn is confirmed [22][5].
10. **Truncation needs playback position plus output latency.** Transcripts don't align exactly with audio [24][26].
11. **Double or aggressive noise suppression hurts STT and turn detection** [89].
12. **OS echo cancellation varies by device.** Windows depends on the hardware vendor and may skip USB/Bluetooth; macOS voice processing has setup-order traps. Test on real hardware [28][29].
13. **Vendor benchmarks crown their own vendor.** Ship defaults backed by our own runs [11][12].
14. **Open full-duplex models over-take the floor.** Wrap them in the same echo and false-interruption guards, or mark them experimental [31][32].

## 9. Sources

1. Silero VAD releases (v5.0 2024-06-27 … v6.0 2025-08-26, v6.2 2025-11-06, v6.2.1 2026-02-24, v6.2.2 2026-09-17, v6.2.3 2026-09-23) — GitHub — <https://github.com/snakers4/silero-vad/releases>
2. snakers4/silero-vad — README (GitHub) — accessed 2026-09-24 — <https://github.com/snakers4/silero-vad>
3. silero-vad 6.2.3 on PyPI (wheel contents, dependencies) — 2026-09-23 — <https://pypi.org/project/silero-vad/>
4. pipecat-ai/smart-turn — README (v3.2, BSD-2) — GitHub — <https://github.com/pipecat-ai/smart-turn>
5. Pipecat src/pipecat/turns/ (user_turn_strategies, speech_timeout/eager/min_words strategies, speculation_gate, user_turn_completion_mixin) and audio/turn/smart_turn/base_smart_turn.py — commit 9d4c508, 2026-09-23 — <https://github.com/pipecat-ai/pipecat/tree/main/src/pipecat/turns>
6. LiveKit — Solving end-of-turn detection: LiveKit Turn Detector v1.0 — 2026-06-17 — <https://livekit.com/blog/solving-end-of-turn-detection>
7. Deepgram — Introducing Flux: Conversational Speech Recognition — Oct 2025 — <https://deepgram.com/learn/introducing-flux-conversational-speech-recognition>
8. AssemblyAI docs — Turn detection (Universal-Streaming) — accessed 2026-09-24 — <https://www.assemblyai.com/docs/streaming/universal-streaming/turn-detection>
9. Kyutai — Kyutai STT (semantic VAD, flush trick) — accessed 2026-09-24 — <https://kyutai.org/stt/>
10. Gradium — Semantic VAD for voice agents: turn detection 2026 — 2026-06-30, updated 2026-09-10 — <https://gradium.ai/content/semantic-vad-voice-agents-turn-detection-2026>
11. livekit/eot-bench — README and results (Apache-2.0) — GitHub, accessed 2026-09-24 — <https://github.com/livekit/eot-bench>
12. Krisp — Turn-Taking and Interruption Prediction in Voice AI (Turn Prediction v3, Interruption Prediction v1) — May 2026 — <https://krisp.ai/blog/voice-ai-turn-taking-interruption-prediction/>
13. LiveKit Model License Agreement — last updated 2024-11-25 — <https://huggingface.co/livekit/turn-detector/blob/main/LICENSE>
14. ten-vad LICENSE (Apache-2.0 with additional conditions) — GitHub — <https://github.com/TEN-framework/ten-vad/blob/main/LICENSE>
15. ten-turn-detection LICENSE (Apache-2.0 with additional restrictions) — GitHub — <https://github.com/TEN-framework/ten-turn-detection/blob/main/LICENSE>
16. vogent/Vogent-Turn-80M — model card (modified Apache-2.0) — Hugging Face — <https://huggingface.co/vogent/Vogent-Turn-80M>
17. videosdk-live/Namo-Turn-Detector-v1-Multilingual — model card — Hugging Face, 2025 — <https://huggingface.co/videosdk-live/Namo-Turn-Detector-v1-Multilingual>
18. LiveKit docs — Turn detector — accessed 2026-09-24 — <https://docs.livekit.io/agents/logic/turns/turn-detector/>
19. Pipecat docs — Smart Turn overview — accessed 2026-09-24 — <https://docs.pipecat.ai/api-reference/server/utilities/turn-detection/smart-turn-overview>
20. OpenAI API reference — Realtime client events (turn_detection, conversation.item.truncate) — accessed 2026-09-24 — <https://developers.openai.com/api/reference/resources/realtime/client-events>
21. LiveKit Agents source — voice/turn.py, voice/endpointing.py, voice/agent_activity.py, voice/audio_recognition.py, voice/events.py, inference/interruption.py, inference/eot/ — commit bf92dd0, 2026-09-23 — <https://github.com/livekit/agents/tree/main/livekit-agents/livekit/agents>
22. Deepgram docs — Optimize voice agent latency with Eager End of Turn — accessed 2026-09-24 — <https://developers.deepgram.com/docs/flux/voice-agent-eager-eot>
23. LiveKit — Solving unwanted interruptions with Adaptive Interruption Handling — 2026-03-19 — <https://livekit.com/blog/adaptive-interruption-handling>
24. OpenAI — Realtime conversations guide (interruptions and truncation) — accessed 2026-09-24 — <https://developers.openai.com/api/docs/guides/realtime-conversations>
25. LiveKit Python SDK — livekit-rtc/livekit/rtc/apm.py, media_devices.py, platform_audio.py — commit ee527bd, 2026-09-23 — <https://github.com/livekit/python-sdks/tree/main/livekit-rtc/livekit/rtc>
26. LiveKit Agents console audio I/O with WebRTC APM (livekit-agents/livekit/agents/cli/_legacy.py) — commit bf92dd0, 2026-09-23 — <https://github.com/livekit/agents/blob/main/livekit-agents/livekit/agents/cli/_legacy.py>
27. livekit 1.1.20 on PyPI (wheel platforms, Apache-2.0) — 2026-09-23 — <https://pypi.org/project/livekit/>
28. Microsoft Learn — Windows 11 APIs for Audio Processing Objects (Acoustic Echo Cancellation) — updated 2025-07-18 — <https://learn.microsoft.com/en-us/windows-hardware/drivers/audio/windows-11-apis-for-audio-processing-objects>
29. Barock — Why your iOS voice agent still hears itself: VoiceProcessingIO, AEC tail, and why 300ms isn't enough — 2026-04-22 — <https://barock.dev/2026/04/22/why-your-ios-voice-agent-still-hears-itself>
30. onnxruntime on PyPI — release/wheel matrix 1.23.2 (2025-10-22) … 1.30.0 (2026-09-10) — queried 2026-09-24 — <https://pypi.org/project/onnxruntime/>
31. Lin et al. — Full-Duplex-Bench (ASRU 2025; arXiv 2503.04721), Table III — 2025-03-06 — <https://arxiv.org/html/2503.04721>
32. Full-Duplex Interaction in Spoken Dialogue Systems: A Comprehensive Study from the ICASSP 2026 HumDial Challenge (arXiv 2604.21406v2) — 2026 — <https://arxiv.org/html/2604.21406v2>
33. Silero VAD src/silero_vad/utils_vad.py (window sizes, context, default thresholds) — master, 2026-09 — <https://github.com/snakers4/silero-vad/blob/master/src/silero_vad/utils_vad.py>
34. Silero VAD wiki — Performance Metrics — GitHub wiki — <https://github.com/snakers4/silero-vad/wiki/Performance-Metrics>
35. TEN-framework/ten-vad — README — accessed 2026-09-24 — <https://github.com/TEN-framework/ten-vad>
36. ten-vad 1.0.6.8 on PyPI (bundled native libraries) — 2025-11-14 — <https://pypi.org/project/ten-vad/>
37. wiseman/py-webrtcvad — GitHub — <https://github.com/wiseman/py-webrtcvad>
38. webrtcvad-wheels 2.0.14 on PyPI — 2024-09-05 — <https://pypi.org/project/webrtcvad-wheels/>
39. Picovoice/cobra (v3.0.0 released 2025-12-12) — GitHub — <https://github.com/Picovoice/cobra>
40. Picovoice Cobra VAD documentation — Picovoice — <https://picovoice.ai/docs/cobra/>
41. nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0 — model card — Hugging Face — <https://huggingface.co/nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0>
42. Pipecat src/pipecat/audio/vad/vad_analyzer.py (VADParams defaults, VAD state machine) — commit 9d4c508, 2026-09-23 — <https://github.com/pipecat-ai/pipecat/blob/main/src/pipecat/audio/vad/vad_analyzer.py>
43. Pipecat src/pipecat/audio/ (vad/aic_quail_vad.py, vad/krisp_viva_vad.py, filters/) — commit 9d4c508, 2026-09-23 — <https://github.com/pipecat-ai/pipecat/tree/main/src/pipecat/audio>
44. Silero VAD wiki — Quality Metrics (v6 vs v5 vs TEN VAD vs WebRTC) — GitHub wiki — <https://github.com/snakers4/silero-vad/wiki/Quality-Metrics>
45. LiveKit Agents Silero plugin (vad.py defaults, onnx_model.py session options) — commit bf92dd0, 2026-09-23 — <https://github.com/livekit/agents/tree/main/livekit-plugins/livekit-plugins-silero/livekit/plugins/silero>
46. Google — Gemini Live API reference (AutomaticActivityDetection, ActivityHandling, TurnCoverage) — accessed 2026-09-24 — <https://ai.google.dev/api/live>
47. Daily — Announcing Smart Turn v3, with CPU inference in just 12ms — 2025-09-11 — <https://www.daily.co/blog/announcing-smart-turn-v3-with-cpu-inference-in-just-12ms/>
48. Daily — Improved accuracy in Smart Turn v3.1 — 2025-12-03 — <https://www.daily.co/blog/improved-accuracy-in-smart-turn-v3-1/>
49. Daily — Smart Turn v3.2: handling noisy environments and short responses — 2026-01-07 — <https://www.daily.co/blog/smart-turn-v3-2-handling-noisy-environments-and-short-responses/>
50. pipecat-ai/smart-turn-v3 — model card and ONNX files — Hugging Face — <https://huggingface.co/pipecat-ai/smart-turn-v3>
51. livekit/turn-detector — model card (text model) — Hugging Face, last modified 2026-02-11 — <https://huggingface.co/livekit/turn-detector>
52. TEN-framework/ten-turn-detection — README — GitHub — <https://github.com/TEN-framework/ten-turn-detection>
53. vogent/vogent-turn (v0.1.0 2025-10-19) — GitHub — <https://github.com/vogent/vogent-turn>
54. Vogent — VoTurn-80M: state-of-the-art turn detection for voice agents — 2025-10-17 — <https://blog.vogent.ai/posts/voturn-80m-state-of-the-art-turn-detection-for-voice-agents>
55. fixie-ai/ultraVAD — model card — Hugging Face — <https://huggingface.co/fixie-ai/ultraVAD>
56. Ultravox — UltraVAD is now open source — date unverified — <https://www.ultravox.ai/blog/ultravad-is-now-open-source-introducing-the-first-context-aware-audio-native-endpointing-model>
57. Pipecat src/pipecat/turns/user_start/krisp_viva_ip_user_turn_start_strategy.py — commit 9d4c508, 2026-09-23 — <https://github.com/pipecat-ai/pipecat/blob/main/src/pipecat/turns/user_start/krisp_viva_ip_user_turn_start_strategy.py>
58. Deepgram docs — Flux end-of-turn detection parameters — accessed 2026-09-24 — <https://developers.deepgram.com/docs/flux/configuration>
59. Deepgram docs — Getting started with Flux — accessed 2026-09-24 — <https://developers.deepgram.com/docs/flux/quickstart>
60. OpenAI — Voice activity detection (Realtime API guide) — accessed 2026-09-24 — <https://developers.openai.com/api/docs/guides/realtime-vad>
61. Google — Gemini Live API capabilities guide (VAD configuration) — accessed 2026-09-24 — <https://ai.google.dev/gemini-api/docs/live-api/capabilities>
62. livekit/eot-bench-data — dataset card (CC-BY-4.0) — Hugging Face, 2026-06/07 — <https://huggingface.co/datasets/livekit/eot-bench-data>
63. TamilEOT: A Dataset and Model for Semantic End-of-Turn Detection in Tamil Telephone Speech (arXiv 2609.05631) — 2026-09-04 — <https://arxiv.org/abs/2609.05631>
64. Inoue, Jiang, Ekstedt, Kawahara, Skantze — Real-time and Continuous Turn-taking Prediction Using Voice Activity Projection (arXiv 2401.04868) — 2024-01-10 — <https://arxiv.org/abs/2401.04868>
65. Ekstedt & Skantze — TurnGPT (Findings of EMNLP 2020; arXiv 2010.10874) — 2020-10-21 — <https://arxiv.org/abs/2010.10874>
66. Li et al. — Easy Turn: Integrating Acoustic and Linguistic Modalities for Robust Turn-Taking (arXiv 2509.23938) — 2025-09-28 — <https://arxiv.org/abs/2509.23938>
67. Tsoi et al. — Next-Turn: Duration-Aware Streaming Endpoint Detection (arXiv 2606.18094) — 2026-06-16 — <https://arxiv.org/abs/2606.18094>
68. Xia et al. — Semantic-Aware Interruption Detection in Spoken Dialogue Systems: Benchmark, Metric, and Model (arXiv 2603.24144) — 2026-03-25 — <https://arxiv.org/abs/2603.24144>
69. Arora et al. — Talking Turns: Benchmarking Audio Foundation Models on Turn-Taking Dynamics (ICLR 2025; arXiv 2503.01174) — 2025-03-03 — <https://arxiv.org/abs/2503.01174>
70. Deepgram docs — Understanding the Flux state machine — accessed 2026-09-24 — <https://developers.deepgram.com/docs/flux/state>
71. AssemblyAI — Voice agent turn detection — 2026-08-25 — <https://www.assemblyai.com/blog/voice-agent-turn-detection>
72. LiveKit docs — Adaptive interruption handling — accessed 2026-09-24 — <https://docs.livekit.io/agents/logic/turns/adaptive-interruption-handling/>
73. LiveKit — Configuring turn detection and interruptions in LiveKit Agents — 2026-06-30 — <https://livekit.com/blog/turn-detection-and-interruption-handling>
74. Pipecat src/pipecat/services/openai/realtime/llm.py (_truncate_current_audio_response) — commit 9d4c508, 2026-09-23 — <https://github.com/pipecat-ai/pipecat/blob/main/src/pipecat/services/openai/realtime/llm.py>
75. Salimi et al. — IHBench: Evaluating Post-Interruption Recovery in Voice Agents (arXiv 2606.19595) — 2026-06-17 — <https://arxiv.org/abs/2606.19595>
76. WebRTC.ventures — How Voice AI Agents Handle Interruption: State Machines vs Streaming Approaches — 2026-09-09 — <https://webrtc.ventures/2026/09/voice-ai-interruption-handling-state-machines-vs-streaming/>
77. aec-audio-processing 1.0.1 on PyPI (WebRTC AudioProcessing 2 via SWIG) — 2025-09-01 — <https://pypi.org/project/aec-audio-processing/>
78. xiongyihui/python-webrtc-audio-processing — GitHub — <https://github.com/xiongyihui/python-webrtc-audio-processing>
79. webrtc-audio-processing 0.1.3 on PyPI — 2019-05-27 — <https://pypi.org/project/webrtc-audio-processing/>
80. pyaec 1.0.1 on PyPI (bindings to aec-rs) — 2024-12-08 — <https://pypi.org/project/pyaec/>
81. thewh1teagle/aec-rs — Acoustic echo cancellation in Rust based on SpeexDSP — GitHub — <https://github.com/thewh1teagle/aec-rs>
82. speexdsp 0.1.1 on PyPI — 2018-07-17 — <https://pypi.org/project/speexdsp/>
83. speexdsp-ns 0.1.2 on PyPI — 2023-08-07 — <https://pypi.org/project/speexdsp-ns/>
84. pengzhendong/pyrnnoise — Python wrapper for RNNoise v0.2 (Apache-2.0) — GitHub — <https://github.com/pengzhendong/pyrnnoise>
85. xiph/rnnoise (BSD-3-Clause) — GitHub — <https://github.com/xiph/rnnoise>
86. pyrnnoise 0.4.5 on PyPI — 2026-09-23 — <https://pypi.org/project/pyrnnoise/>
87. Rikorose/DeepFilterNet — README (MIT/Apache-2.0) — GitHub — <https://github.com/Rikorose/DeepFilterNet>
88. deepfilterlib / deepfilternet 0.5.6 on PyPI — 2023-08-31 — <https://pypi.org/project/deepfilterlib/>
89. LiveKit docs — Noise & echo cancellation — accessed 2026-09-24 — <https://docs.livekit.io/transport/media/noise-cancellation/>
90. NVIDIA Audio Effects (AFX) SDK — Noise Removal / Background Noise Suppression effect — accessed 2026-09-24 — <https://docs.nvidia.com/maxine/afx/latest/AboutTheEffects/AboutNoiseRemovalBackgroundNoiseSuppression.html>
91. [ANNOUNCE] WebRTC AudioProcessing v2.0 (pulseaudio-discuss) — 2025-01-08 — <https://www.mail-archive.com/pulseaudio-discuss@lists.freedesktop.org/msg22107.html>
92. Apple Developer — AVAudioIONode.setVoiceProcessingEnabled(_:) (macOS 10.15+) — accessed 2026-09-24 — <https://developer.apple.com/documentation/avfaudio/avaudioionode/setvoiceprocessingenabled(_:)>
93. Apple Developer — kAudioUnitSubType_VoiceProcessingIO — accessed 2026-09-24 — <https://developer.apple.com/documentation/audiotoolbox/kaudiounitsubtype_voiceprocessingio>
94. pyobjc-framework-AVFoundation 12.2.2 on PyPI — 2026-08-11 — <https://pypi.org/project/pyobjc-framework-AVFoundation/>
95. Microsoft Learn — Audio Signal Processing Modes — updated 2025-08-08 — <https://learn.microsoft.com/en-us/windows-hardware/drivers/audio/audio-signal-processing-modes>
96. Microsoft Windows-classic-samples — AcousticEchoCancellation README — GitHub — <https://github.com/microsoft/Windows-classic-samples/blob/main/Samples/AcousticEchoCancellation/README.md>
97. PyAudioWPatch 0.2.12.8 on PyPI (PortAudio fork with WASAPI loopback) — 2026-01-14 — <https://pypi.org/project/pyaudiowpatch/>
98. PipeWire documentation — Echo-cancel module — accessed 2026-09-24 — <https://docs.pipewire.org/page_module_echo_cancel.html>
99. Nguyen et al. — Generative Spoken Dialogue Language Modeling (dGSLM; arXiv 2203.16502) — 2022-03-30 — <https://arxiv.org/abs/2203.16502>
100. Moshi: a speech-text foundation model for real-time dialogue (arXiv 2410.00037) — 2024-09-17 — <https://arxiv.org/abs/2410.00037>
101. kyutai-labs/moshi — README (licenses, Mimi, latency, backends) — GitHub — <https://github.com/kyutai-labs/moshi>
102. Veluri et al. — Beyond Turn-Based Interfaces: Synchronous LLMs as Full-Duplex Dialogue Agents (SyncLLM; EMNLP 2024) — 2024-09-23 — <https://arxiv.org/abs/2409.15594>
103. Freeze-Omni: A Smart and Low Latency Speech-to-speech Dialogue Model with Frozen LLM (arXiv 2411.00774) — 2024-11-01 — <https://arxiv.org/abs/2411.00774>
104. OmniFlatten: An End-to-end GPT Model for Seamless Voice Conversation (arXiv 2410.17799) — 2024-10-23 — <https://arxiv.org/abs/2410.17799>
105. SALMONN-omni: A Standalone Speech LLM without Codec Injection for Full-duplex Conversation (arXiv 2505.17060) — 2025-05-17 — <https://arxiv.org/abs/2505.17060>
106. Roy et al. — PersonaPlex: Voice and Role Control for Full Duplex Conversational Speech Models (arXiv 2602.06053) — 2026-01-14 — <https://arxiv.org/abs/2602.06053>
107. NVIDIA ADLR — PersonaPlex — 2026-01-15 — <https://research.nvidia.com/labs/adlr/personaplex>
108. NVIDIA/personaplex — README (MIT code, NVIDIA Open Model License weights) — GitHub — <https://github.com/NVIDIA/personaplex>
109. SoulX-Duplug: Plug-and-Play Streaming State Prediction Module for Realtime Full-Duplex Speech Conversation (arXiv 2603.14877) — 2026-03-16 — <https://arxiv.org/abs/2603.14877>
110. Full-Duplex-Bench GitHub (v1.0 2025-03, v1.5 2025-08, v2.0 2026-02, v3.0 2026-05) — GitHub — <https://github.com/DanielLin94144/Full-Duplex-Bench>
111. A Survey of Full-Duplex Spoken Dialogue Systems: Architectural Hierarchy, Interaction Ontology, and Decision State Machine (arXiv 2606.19453) — 2026-06 — <https://arxiv.org/html/2606.19453v1>
112. From Turn-Taking to Synchronous Dialogue: A Survey of Full-Duplex Spoken Language Models (arXiv 2509.14515) — 2025-09 — <https://arxiv.org/html/2509.14515v1>
113. pipecat-ai/smart-turn-v2 — model card — Hugging Face — <https://huggingface.co/pipecat-ai/smart-turn-v2>
114. sounddevice 0.5.6 on PyPI (MIT) — 2026-08-17 — <https://pypi.org/project/sounddevice/>
