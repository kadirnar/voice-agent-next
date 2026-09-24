# 05: Latency Engineering, Transports, Telephony, Deployment and Production

*Research note for **voice-agent-next**. Compiled 2026-09-24. Sources are listed in §10 and cited inline as [n]. Vendor-run benchmarks are labelled "vendor-reported". Figures I could not confirm from a primary source are marked "(unverified)". Figures I computed from cited numbers are marked "(derived)".*

---

## 1. Executive summary

- **Targets.** Aim for about 800 ms voice-to-voice (V2V), expect 1.1–1.3 s on typical cloud cascades [1][2], and treat over 1.5 s as broken. Humans leave gaps of about 200 ms but need over 600 ms to plan speech, so they predict turn ends [58]. Open models on a local RTX 5090 reach 508 ms P50 [75].
- **Where the time goes.** STT with endpointing plus LLM time-to-first-token (TTFT) take about 73% of the primer's 1,293 ms budget [1][3] (derived). Device I/O, codec, network and jitter buffers cost about 200 ms per round trip on WebRTC [1] (derived). The PSTN adds roughly 230 ms: Twilio targets 1,115 ms mouth-to-ear against 885 ms in-platform [2] (derived).
- **Biggest levers.**
  - Stream every stage.
  - Semantic turn detection: Smart Turn v3 takes about 12 ms on CPU [45]; Flux claims 200–600 ms savings [49].
  - Speculative LLM start on an "eager" end of turn, at a cost of 50–70% more LLM calls [48].
  - Send the first clause to TTS immediately.
  - Warm connections, regional colocation, prompt caching.
- **Measure acoustically.** Component sums do not match the gap users hear [65][66]. Server-side observers miss client playout [7]. TTFB instrumentation has had bugs [64]. Ground truth is a recording measured from end of user speech to start of agent audio [1][85].
- **Transport.** WebRTC for any client that crosses the internet [9][22]. WebSocket for server-to-server links and telephony media streams. SIP for phones; OpenAI and LiveKit accept SIP directly [20][26].
- **Telephony.** Each provider has its own WebSocket dialect. Twilio is μ-law 8 kHz only [13]; Telnyx, Vonage and Plivo offer 16 kHz linear PCM [15][17][18]. Normalize them behind one serializer interface with clear, mark and DTMF capabilities.
- **Local audio.** `sounddevice` (PortAudio) is the default; its wheels bundle PortAudio on Windows and macOS [29]. Debian and Ubuntu's PortAudio has no PulseAudio/PipeWire host API [30]. Nothing in PortAudio or miniaudio cancels echo, so ship WebRTC APM (10 ms frames plus a reverse stream [115]) or a half-duplex mode.
- **Deployment.** One CPU orchestrator per session, from a prewarmed process pool with CPU-load admission; LiveKit defaults to idle processes = CPU count and a 0.7 load threshold [40]. Run GPU inference as separate autoscaled services [8]. Cold starts take seconds (Pipecat Cloud: about 10 s floor [42]), so keep warm capacity.
- **Reliability.** Failover must also catch silent failures. In one Pipecat incident a silent reconnect caused 66 s of dead air and pushed unspoken text into the LLM context [63]. Only commit text the user actually heard.
- **Observability.** OpenTelemetry spans nested as conversation → turn → STT/LLM/TTS are the de facto schema [6]. Standard realtime-voice conventions are still an open PR [70].
- **Costs.** One vendor's fleet data: cascaded about $0.07–0.13/min all-in, speech-to-speech $0.18–0.21/min [89] (vendor-reported). GPT-Live's voice layer costs $0.05/min plus the backend model [24]. Self-hosting pays off only at sustained high utilization [90].
- **On-device.**
  - Silero VAD takes under 1 ms per chunk [83].
  - Smart Turn is 8 MB at int8 [45]; Kokoro ONNX is 80 MB quantized [82].
  - whisper.cpp and sherpa-onnx run on all three desktop OSes, and MLX covers Apple Silicon [80][84][78].

---

## 2. Latency

### 2.1 What "natural" means

- **Human turn-taking.**
  - Across 10 languages, speakers avoid overlap and minimize silence; average gaps stay within 250 ms of the cross-language mean [57].
  - Gaps are about 200 ms, but producing speech takes over 600 ms, so listeners plan their reply before the turn ends [58].
  - An agent therefore has to start work before the turn ends: speculation is a requirement, not an optional optimization.
- **Network planning.** ITU-T G.114: one-way delay up to 150 ms does not significantly affect most applications; 400 ms is the planning ceiling [59]. This applies per transport leg.
- **Industry thresholds.**

  | Source | Threshold |
  |---|---|
  | Primer [1] | About 500 ms is typical for human responses; 1,500 ms V2V is "an important target to aim for" |
  | WebRTC.ventures [3] | About 800 ms before a conversation starts to feel slow |
  | Coval [88] | Under 500 ms feels real-time; under 800 ms is acceptable; over 1,200 ms feels broken |
  | Cresta [85] | Pauses of about 300 ms can already feel unnatural; the experience degrades quickly beyond about 1.5 s |

### 2.2 Reference latency budgets

**Table 1. Stage-by-stage budget for a WebRTC client on macOS [1]**

| Stage | ms | Tunable? |
|---|---|---|
| Mic input (macOS) | 40 | Device and OS |
| Opus encode + network + packet handling + jitter buffer + decode (inbound) | 74 | Transport |
| Transcription and endpointing | 300 | **Yes: biggest lever after the LLM** |
| LLM time to first byte | 650 | **Yes: model, region, caching** |
| Sentence aggregation | 20 | Yes |
| TTS time to first byte | 120 | Yes |
| Encode + network + jitter buffer + decode (outbound) | 74 | Transport |
| Speaker output | 15 | Device |
| **Total** | **1,293** | |

**Table 2. Twilio's budget for a PSTN voice agent (target / upper limit) [2]**

| Metric | Target | Upper |
|---|---|---|
| Mouth-to-ear turn gap (what the user perceives) | 1,115 ms | 1,400 ms |
| Platform turn gap (excluding the public network) | 885 ms | 1,100 ms |
| Speech-to-text | 350 ms | 500 ms |
| LLM TTFT | 375 ms | 750 ms |
| TTS time to first byte | 100 ms | 250 ms |

Twilio also itemizes fixed costs: 40 ms of audio network ingress, 30 ms of buffering, 25 ms of decoding, and about 95 ms of service hops, re-encoding and buffering [2].

**Table 3. Published end-to-end measurements**

| System | Result | Notes |
|---|---|---|
| Nemotron ASR + Nemotron 3 Nano + Magpie TTS, RTX 5090 [75] | V2V **P50 508 / P90 544 ms**; ASR 19, LLM 171, TTS 108 ms (P50) | Local, server-side |
| Same stack, DGX Spark [75] | V2V P50 1,180 / P90 1,359 ms | Hardware matters |
| NVIDIA cache-aware ASR, full pipeline [74] | V2V under 900 ms; final transcript 24 ms (median) | Local |
| Modal + Pipecat + open models [8] | Median V2V about 1 s | WebRTC, one region |
| Kyutai Unmute [72] | TTS latency 750 ms (one L40S) vs 450 ms (separate GPUs) | |
| Twilio ConversationRelay [2] | Median under 0.5 s, P95 about 0.725 s | Platform-side, vendor-reported |
| DestiLabs, 12 telephony projects [89] | 680 ms P50 / 1,180 ms P95; speech-to-speech 540–580 ms P50; cascaded 610–810 ms P50 | Vendor-reported |
| "Tested Media" study, quoted by Telnyx [87] | Vapi 720/1,050 ms; Retell 680/920 ms; Bland 850/1,180 ms (median/P95) | Vendor-reported, (unverified) |
| SignalWire [86] | 900–1,500 ms in production; 2–3 s "typical" elsewhere | Vendor-reported |

### 2.3 Component reference numbers

- **STT.**
  - Deepgram's guidance for streaming: expect 20–200 ms of network transit, under 300 ms of transcription, and 200–500 ms end to end [50].
  - Nemotron Speech ASR delivers a final transcript 24 ms after speech ends (median, local GPU) [74].
  - Kyutai's STT model runs with a fixed 500 ms delay (1B model) [73].
- **Turn detection.**
  - Smart Turn v3 inference on AWS CPUs: 12.6 ms (c7a.2xlarge), 59.8 ms (c8g.medium), 94.8 ms (t3.medium) [45]. On GPU it takes 1–5 ms [46].
  - The primer's layered approach (VAD, then Smart Turn, then an LLM tag) decides about 250 ms after the user pauses [1].
  - Deepgram Flux end-of-turn detection: P90 about 1 s, P95 about 1.5 s. `eot_threshold` defaults to 0.7, the recommended eager threshold is 0.3–0.5, and the silence fallback defaults to 5,000 ms [49].
  - Silence-timeout defaults elsewhere: Vapi `waitSeconds` 0.4 s [93]; Pipecat recommends VAD `stop_secs` 0.2 s with a 3.0 s Smart Turn fallback [47].
- **LLM TTFT.**
  - Primer, median/P95 in ms: GPT-4.1 536/1,771; Gemini 2.5 Flash 597/1,137; Claude 4.5 Haiku 637/1,615; GPT-5.1 739/1,492; Nemotron 3 Ultra self-hosted 541/712 [1]. The primer treats 600 ms or less as fast enough [1]. The self-hosted model's tight P95 is the notable result.
  - Artificial Analysis data at 10k input tokens, median: gpt-oss-120b on Baseten 0.23 s and on Cerebras 0.49 s; OpenAI GPT-5.6 Luna 0.74 s; Gemini 3.5 Flash 0.90 s [53]. The same article argues the metric that matters for voice is time to first *sentence*, not first token [53].
- **TTS time to first audio (median/P95, ms, cost per minute) [1].**
  - Cartesia Sonic 3.5: 195/240, $0.028
  - Gradium: 235/320, $0.032
  - Deepgram Aura-2: 310/600, $0.024
  - ElevenLabs Turbo v2.5: 330/670, $0.050
  - Inworld TTS 1.5 Max: 337/560, $0.009

  ElevenLabs' "~75 ms" for Flash measures model inference only; network adds 20–200 ms [51].

### 2.4 How to measure V2V correctly

1. **Definition.** V2V is the time from the end of the user's speech (acoustic) to the start of agent audio at the user's ear. Twilio calls this "mouth-to-ear" and separates it from the "platform turn gap" [2]. SignalWire's rule: "end of your speech → audio playback. Anything less is partial accounting" [86].
2. **Ground truth.** Record the conversation and measure the gap on the waveform [1]. To get a distribution, place simulated calls over the real media path, record the caller side, and extract gaps with ASR [85].
3. **Server-side proxies undercount.**
   - Pipecat's `UserBotLatencyObserver` is server-side only. It backdates end of speech by VAD `stop_secs`, and its per-stage `contributions` sum to the measured interval [7].
   - LiveKit's formula `end_of_utterance_delay + llm.ttft + tts.ttfb` was off from measured silence by 0.3–0.7 s in one user report involving tool calls and sentence tokenization [65]. It also omits `on_user_turn_completed_delay` [66].
4. **Instrument per utterance, not per connection.** Pipecat's WebSocket TTS services recorded TTFB only on the first request of a session, because the timer was tied to the lifetime of the `context_id` [64].
5. **Measuring STT.** Latency = audio cursor − transcript cursor, computed on *interim* results. Final results include endpointing delay, and provider timestamps are not accurate to the millisecond. Stream audio in 20–100 ms chunks [50]. Pipecat's STT benchmark paces synthetic audio in real time and reports TTFS (time from end of speech to final transcript) [91].
6. **Statistics.** Report P50/P90/P95/P99, split out first-turn latency, and measure under production concurrency [88]. Track greeting latency (connect → first bot speech) as its own metric [7].

### 2.5 Optimization techniques

1. **Stream every stage.** This is the highest-leverage change [3], and TTS must accept text incrementally. Cartesia's WebSocket has a `continue` flag and buffers text for up to `max_buffer_delay_ms` (default 3,000 ms) before generating [52]. Frameworks must therefore flush at clause boundaries or lower that limit.
2. **Turn detection.** A fixed 800 ms silence timeout "adds nearly a full second to every single response" [96]. Semantic models bring detection below 300 ms, compared with a VAD-only baseline of 600 ms or more [85].
3. **Speculation.**
   - LiveKit's preemptive generation starts the LLM on partial transcripts. LiveKit rates the benefit as "dependent on circumstances" [4][94]: whenever the final transcript differs, the speculative tokens are wasted.
   - Deepgram's `EagerEndOfTurn` / `TurnResumed` / `EndOfTurn` events support speculative replies that are cancelled when the user resumes. Deepgram estimates 50–70% more LLM calls and "hundreds of milliseconds" saved [48].
4. **First-sentence fast path.** Send the first speakable clause to TTS immediately. Tool calls are slow: at 450 ms TTFB and 100 tokens/s, emitting a 100-token function call takes 1,450 ms before the tool even runs [1].
5. **Prompt caching.**
   - OpenAI caches prompts of 1,024+ tokens automatically on GPT-5.6+, with up to 90% off cached input [54]. TTFT is 7% faster at 1,024 tokens and 67% faster at 150k+ [55]. The Realtime API's `retention_ratio` keeps the prefix stable [55].
   - Anthropic uses `cache_control` markers: 5-minute TTL by default (1 hour optional), a 512–4,096-token minimum depending on the model, and `max_tokens: 0` requests to pre-warm [56].
   - Context is re-sent every turn, so LLM cost grows superlinearly with call length. Gemini 2.5 Flash costs $0.002 for a 3-minute call and $0.024 for a 30-minute call [1].
6. **Warm connections.**
   - Use persistent keep-alive connections [2], reuse connections, and keep DNS lookups off the critical path [85].
   - Prewarming VAD is "table stakes" [4].
7. **Colocation and regions.**
   - Colocate STT, LLM, TTS and media servers, and put SIP trunks near the agent [2][3][4].
   - Singapore ↔ US-East round trips take 230–280 ms in practice [9]. Serving Australia adds about 200–300 ms [85].
   - One LiveKit Cloud EU deployment went from about 2 s per turn to over 4 s because its providers were far away [67].
   - Use regional endpoints: ElevenLabs routes to clusters in North America, Europe and South-East Asia [51], and OpenAI has an EU SIP endpoint [20].
8. **No needless transcoding.** Match sample rates: 16 kHz for STT, 24–48 kHz for TTS [2]. Telnyx now decouples stream codecs from call codecs, which removes a transcoding step [15].
9. **Hedging and routing.** Fire parallel LLM calls and take whichever answers first [85]. Route turns to models by query complexity [3].
10. **Fillers.**
    - Start a watchdog timer and speak a "one moment" message only if a tool is still running when it fires. Background music is another option [1].
    - LiveKit's `BackgroundAudioPlayer` plays a `thinking_sound` such as `KEYBOARD_TYPING` [97]. Cresta plays wait messages when a call exceeds 1 s [85].
11. **GPU placement.** Interleaving LLM and TTS on a single GPU helps local deployments [75]. Splitting STT, LLM and TTS across GPUs cut Unmute's TTS latency from 750 to 450 ms [72].

One LiveKit-focused guide claims that stacking about a dozen of these techniques cuts P95 turn latency from 1.2–1.4 s to 500–650 ms [112] (vendor-reported, unverified).

---

## 3. Transports

### 3.1 Transport comparison

| | WebRTC (UDP/SRTP) | WebSocket (TCP) | WebTransport (QUIC) | SIP/RTP |
|---|---|---|---|---|
| **Loss behavior** | A lost 20 ms frame is concealed; no stall [9] | Head-of-line blocking stalls playback for "hundreds of milliseconds" [9] | No head-of-line blocking [10] | UDP; carrier-grade |
| **Media timing** | Adaptive jitter buffer, timestamps, GCC congestion control [9] | None; you build pacing and jitter handling yourself [9] | Build it yourself | Built into RTP stacks |
| **Codec** | Opus: 6–510 kb/s, 2.5–60 ms frames, 8–48 kHz, PLC [109]; about 32 kb/s target, about 20 kb/s average from clients [10] | Usually base64 PCM, about 512 kb/s, roughly 10× Opus [10] | App-defined | G.711 (μ-law/A-law), G.722, Opus, AMR-WB |
| **Echo, noise, gain** | Browser and native SDKs apply AEC/NS/AGC before sending [9][37] | None | None | Handled by the phone or carrier |
| **NAT traversal** | ICE/STUN/TURN [9] | Plain HTTPS | HTTPS | Trunk configuration |
| **Best use** | Browser and mobile clients; OpenAI recommends it for client apps [21][22] | Server-to-server links, telephony media streams, prototypes | Emerging; Safari only added support in 2026 [10] | Phone calls |

Reconnection logic on raw WebSockets is "quite hard", and WebRTC comes with built-in quality statistics [1].

### 3.2 Python WebRTC options

| Library | Status (Sep 2026) | Notes |
|---|---|---|
| **aiortc** | 1.15.0 (2026-07-13), pure-Python wheel, Python 3.10–3.14 [108] | Opus/PCMU/PCMA, ICE (half-trickle), DTLS-SRTP, SCTP data channels, NACK/PLI [38]. No mention of AEC. Pipecat's `SmallWebRTCTransport` is built on it: one peer per connection, and a process-wide SCTP chunk size setting [39]. |
| **livekit** (rtc) | 1.1.20 (2026-09-23), wheels for Windows x64, macOS x64/arm64, Linux x64/arm64, Python 3.9–3.14 [103] | Includes WebRTC `AudioProcessingModule` (AEC, noise suppression, high-pass filter, AGC). Frames must be exactly 10 ms, and `set_stream_delay_ms` must be set when echo processing is on [115]. `MediaDevices` is built on sounddevice and feeds the APM reverse stream automatically [36]. Room connections require a LiveKit server, either Cloud or self-hosted [102]. |
| **daily-python** | 0.32.0 (2026-08-19), Linux and macOS only, **no Windows wheels** [104] | Joins Daily rooms. |

Deployment notes:

- **Peer-to-peer.** Serverless P2P WebRTC is enough for 1:1 agents. For self-hosted setups it saves 10–100 ms compared with routing through a server. You need an SFU or WebRTC cloud for multi-party sessions or global edge routing [11].
- **P2P in the cloud.** A 2026 AWS deployment hit three problems [12]:
  - It had to strip non-relay ICE candidates and force `turn_only`, because unreachable VPC candidates added seconds to connection setup.
  - Signaling needed session affinity across the multi-step handshake.
  - Cold containers sometimes failed after signaling succeeded, so the client needed retry logic.
- **Clients.** Pipecat ships client SDKs for JS, React, React Native, Swift, Kotlin, C++ and ESP32 [101].
  - Browsers expose `echoCancellation` through `getUserMedia` [110], and AudioWorklet runs low-latency processing off the main thread [111].
  - Firefox's AEC is weak; the primer recommends Chrome and Safari [1].

**Speech-to-speech provider endpoints.**

- **OpenAI Realtime** [21][22]:
  - Browsers exchange SDP with `/v1/realtime/calls`, using an ephemeral key from `/v1/realtime/client_secrets`. Events travel on the `oai-events` data channel.
  - Servers connect over WebSocket, sending `audio/pcm` at 24 or 16 kHz, or `audio/pcmu`/`audio/pcma` at 8 kHz, as base64 inside JSON events.
- **GPT-Live** (available in the API by September 2026) [23][24][25]:
  - Full-duplex, and delegates reasoning to a backend model. OpenAI recommends it as the starting point for new conversational apps [114].
  - Costs $0.05 per minute billed per second, with the backend billed separately.
  - Tier limits: 25–500 concurrent sessions.
  - Transports: WebRTC, WebSocket and SIP.

  voice-agent-next must support two modes: relay (client → our server → provider) and direct-to-provider with a server-side sideband control channel.

### 3.3 WebSocket audio framing

- **Binary or JSON.** Vonage sends raw 16-bit little-endian PCM in binary frames (20 ms, 640 bytes at 16 kHz) and uses JSON text frames for control [17]. Twilio, Telnyx, Plivo and OpenAI put base64 inside JSON [13][16][18][21], which adds 33% overhead (derived: 4 bytes encode 3). Raw 16 kHz 16-bit mono is 256 kb/s (derived).
- **Frame size.** Telephony uses 20 ms frames [17][19]. Streaming STT wants 20–100 ms chunks [50]. WebRTC APM requires 10 ms [115]. OpenAI PCM chunks must have an even byte length [21].
- **Control plane.** A protocol needs at least three primitives:
  - **clear**: flush queued playback on barge-in.
  - **mark/checkpoint**: confirm what has been played.
  - **DTMF**: keypad input.

---

## 4. Telephony

**Table 4. Media-stream interfaces by provider**

| Provider | Framing | Audio | Barge-in / playout tracking | DTMF | Notes |
|---|---|---|---|---|---|
| **Twilio Media Streams** [13][14] | JSON over WebSocket, base64 payload without headers | `audio/x-mulaw`, 8,000 Hz, mono, fixed | Send `clear`; `mark` comes back once playback reaches it | `dtmf` event, inbound only, bidirectional streams only | `<Connect><Stream>` makes it bidirectional, with **one bidirectional stream per call** and the inbound track only |
| **Telnyx** [15][16] | JSON, base64 RTP payloads | PCMU (default), PCMA, G722, OPUS (8/16 kHz), AMR-WB (8/16 kHz), L16 16 kHz (added 2025-09-08); bidirectional sampling rate 8/16/24 kHz | `bidirectionalMode` `mp3` or `rtp` | (see docs) | Stream codec can differ from call codec, so no transcoding [15]; `enableReconnect` defaults to true |
| **Vonage** [17] | **Binary PCM frames** plus JSON control | `audio/l16;rate=8000/16000/24000`; 16 kHz recommended for ASR | `{"action":"clear"}` → `websocket:cleared` | `websocket:dtmf` | Buffer holds 3,072 packets (about 60 s) |
| **Plivo** [18][19][100] | JSON, base64 | `audio/x-l16` at 8/16/24 kHz, `audio/x-mulaw` at 8 kHz; 20 ms frames | `playAudio`, `clearAudio`, checkpoints | dtmf events | `keepCallAlive="true"` stops the call from dropping when the stream ends |
| **OpenAI SIP** [20] | SIP over TLS (port 5061) plus SRTP | Negotiated by SIP | Sideband WebSocket via `call_id` | (not documented) | `sip:$PROJECT_ID@sip.api.openai.com;transport=tls` (EU: `sip-eu`); webhook `realtime.call.incoming`; `accept`/`reject`/`refer`/`hangup` |
| **LiveKit SIP** [26] | SIP participants join rooms | Trunk codecs | Uses room audio | RFC 2833/4733 | Inbound/outbound trunks, dispatch rules, cold (REFER) and warm transfer, TLS/SRTP, Krisp |

**PSTN penalty and audio quality.**

- **Added latency.** The gap between Twilio's mouth-to-ear and platform targets implies about 230 ms for public-network legs [2] (derived). One carrier-leg test reports round trips of 71/118 ms P50/P95 on Telnyx and 89/161 ms on Twilio [87] (vendor-reported, unverified).
- **Narrowband audio.** 8 kHz audio hurts recognition. Request 16 kHz where the provider offers it [15][17][18]. LiveKit's turn detector resamples to 16 kHz internally and has a telephony-tuned variant [95].
- **Noise cancellation.** Apply it on the agent side rather than at the trunk, and never twice [37].
- **Transfers.** Use SIP REFER: LiveKit [26], OpenAI `/refer` [20], or Daily at $0.20 per event [43]. Pipecat's serializers hang up automatically when the pipeline ends; disable that for transfer flows, because TwiML resumes once the WebSocket closes [99].

---

## 5. Local audio I/O across platforms

**Table 5. Python audio I/O libraries**

| Library | Packaging | Latency controls | Caveats |
|---|---|---|---|
| **sounddevice** 0.5.6 (2026-08-17) [105] | Wheels bundle PortAudio on Windows (x86/x64/ARM64) and macOS universal2. Linux needs the system `libportaudio2` [29][105] | `latency='low'/'high'`/seconds; `blocksize=0` is the most robust choice; `extra_settings` for WASAPI (`exclusive`, `auto_convert`, `explicit_sample_format`), Core Audio (`change_device_parameters`, `conversion_quality`) and ASIO [27][28] | Callbacks must never block or allocate [27]. ASIO needs `SD_ENABLE_ASIO` [29] |
| **PyAudio** 0.2.14 (2024-11-23) [32] | Wheels on Windows only (bundled PortAudio 19.7.0, no ASIO). Builds from source on macOS and Linux | Blocking or callback | Development is slow; not a good default |
| **pyminiaudio** (miniaudio 0.11.25) [31] | Wheels for all three OSes | `buffersize_msec` (default 200 ms, too high for duplex) | Backends: WASAPI/DirectSound/WinMM, Core Audio, PulseAudio/ALSA/JACK/sndio/OSS, AAudio/OpenSL, WebAudio |
| **livekit.rtc MediaDevices** [36] | Built on sounddevice | 10 ms APM frames [115] | The only option here with built-in AEC |

**Per-OS notes.**

- **Linux.**
  - Debian Trixie and Ubuntu 25.10 ship PortAudio 19.6.0 from 2016. It exposes only ALSA and OSS host APIs; the PulseAudio host API exists only in unreleased PortAudio git [30].
  - With PipeWire, the ALSA `default` and `pipewire` devices are usually routed through PipeWire (unverified).
  - PipeWire provides a WebRTC-based `echo-cancel` module [34] and a `node.latency` hint such as `1024/48000` [35].
- **Windows.**
  - Shared mode defaults to a 10 ms period, and the audio engine adds about 1.3 ms on Windows 10 and later.
  - Smaller periods need `IAudioClient3` plus driver support. The inbox HD Audio driver handles 128–480 samples (2.66–10 ms at 48 kHz).
  - Exclusive mode bypasses the engine but locks the device for other apps.
  - Communications effects reduce echo and noise but add latency; "raw" mode bypasses OEM processing [33].
- **macOS.** Core Audio's `change_device_parameters` lowers latency but "may disrupt other programs" [28]. Apple's voice-processing I/O unit provides OS-level AEC (unverified).
- **Echo.** Without AEC the bot hears itself and interrupts itself [113]. AEC is latency-sensitive and has to run on the device [1]. AEC plus denoising typically adds 25–50 ms [85].

---

## 6. Deployment and scaling

- **Framework baselines (Sep 2026).** pipecat-ai 1.11.0 requires Python 3.11+ [106]; livekit-agents 1.8.3 supports Python 3.10–3.14 [107].
- **Process model.**
  - LiveKit agent servers register with the LiveKit server and run each job in its own process [102].
  - `num_idle_processes` defaults to `ceil(cpu_count)` (0 in dev mode) and respects cgroup limits. `load_threshold` defaults to 0.7, measured as average CPU over 5 s. Drain timeout is 1 hour [40].
  - Avoid burstable instances (AWS t3/t4g), and call `connect()` early in the entrypoint [41].
- **Managed hosting (Pipecat Cloud).**
  - One session per instance. Cold starts take about 10 s, "a floor, not a promise". Buffer capacity arrives within about 30 s. Idle instances live for 5 minutes. Daily-hosted regions cap at 50 agents [42].
  - Warm P99 start time is under 1 s, in four regions [44].
  - Instance sizes run from 0.5 vCPU/1 GB at $0.01 per minute to 1.5 vCPU/3 GB at $0.03 per minute. Reserved instances cost $0.0005–0.0015 per minute [43].
- **GPU inference as separate services.**
  - Keep stateful bots in CPU containers and scale GPU inference independently, pinned to the same region [8].
  - NVIDIA's blueprint uses NIM and Docker Compose profiles. The all-in-one layout needs about 80 GB of VRAM; single-GPU profiles need at least 28 GiB [76].
  - Unmute needs at least 16 GB of VRAM (LLM 6.1, STT 2.5, TTS 5.3 GB) [72].
  - Streaming ASR density: Nemotron ASR handles about 560 streams per H100 at 320 ms chunks, 3× the baseline [74]. Kyutai STT handles 400 streams on an H100, or 64 on an L40S at 3× real time [73].
- **Measure TTFT under load.** In a 2024 benchmark, TensorRT-LLM's TTFT on 70B Q4 exceeded 6 s at 100 users, while vLLM had the best TTFT for 8B [77].

**Table 6. Cost per minute**

| Line item | $/min | Source |
|---|---|---|
| Agent hosting (Pipecat Cloud, active) | 0.01–0.03 | [43] |
| Daily 1:1 WebRTC voice / PSTN / SIP | free / 0.018 / 0.003–0.02 | [43] |
| Plivo SIP plus audio streaming | from 0.0028 | [19] |
| TTS | 0.009–0.050 | [1] |
| LLM, Gemini 2.5 Flash: $0.002 per 3-min call → $0.024 per 30-min call | about 0.0007 → 0.0008 | [1] (derived per-minute) |
| LLM, GPT-4.1: $0.019 per 3-min call → $0.318 per 30-min call | about 0.006 → 0.011 | [1] (derived per-minute) |
| GPT-Live voice layer (backend extra) | 0.05 | [24] |
| All-in, cascaded / speech-to-speech | 0.07–0.13 / 0.18–0.21 | [89] vendor-reported |
| Self-hosted STT on an L40S ($0.99/h rental [90]) ÷ 64 streams [73] | about 0.00026 per stream-minute at 100% utilization | derived |

Deepgram argues self-hosting breaks even at about 2,400 audio-hours per month once DevOps time is counted [90] (vendor-reported). The model is sensitive to utilization.

---

## 7. Production concerns

### 7.1 Observability

- **Span schema.**
  - Pipecat nests `conversation` → `turn` → `stt`/`llm`/`tts` spans. Attributes include `turn.number`, `turn.was_interrupted`, `metrics.ttfb` and `gen_ai.*` usage, with OTLP exporters for Langfuse, Jaeger, Datadog and others [6].
  - LiveKit exports OTel spans, session reports and usage [5]. Its metrics include:
    - `end_of_utterance_delay`, `transcription_delay`, `on_user_turn_completed_delay`
    - `ttft`, `prompt_cached_tokens`
    - TTS `ttfb`
    - interruption `detection_delay` and `num_backchannels`
- **Standards.** The GenAI semantic conventions are still in development. Prompts and responses are not captured by default [71]. The realtime voice proposal (`gen_ai.realtime_inference.client`, `gen_ai.user_speech.internal`, `gen_ai.realtime_session.id`, audio token usage) was still an open PR in September 2026 [70].
- **Recording.** Pipecat's `AudioBufferProcessor` records stereo (user on the left channel, bot on the right) and can emit per-turn audio [98].

### 7.2 Testing and simulation

- **Unit tests in text mode.** LiveKit runs text-mode unit tests with LLM judges. They are cheap and deterministic, and audio runs are reserved for turn-taking issues [60].
- **Commercial simulators.** Coval, Hamming and Cekura run synthetic callers, regression suites and load tests (ramp-up, sustained, spike, soak) [88][92].
- **Latency benchmarks.** Use real-time-paced synthetic input [91] and caller-side recordings [85].

### 7.3 Reliability and failover

- **LiveKit `FallbackAdapter`** (LLM, STT, TTS) [61]:
  - Triggers on connection failures, timeouts, 4xx/5xx errors and mid-stream disconnects.
  - Marks the failed provider unhealthy, probes it in the background, and emits `*_availability_changed` events.
  - TTS never switches mid-utterance once audio has played. LLM fallback after chunks have been sent requires opting in with `retry_on_chunk_sent`.
- **Pipecat `ServiceSwitcher`** [62]: offers manual and failover strategies. Failover fires when a service becomes unusable (bad key, unknown model or voice), and `set_usable(True)` restores it.
- **Case study: Pipecat #5305** [63]. A Cartesia keepalive timeout reconnected silently, with no `ErrorFrame`. The result was 66 s of silence, the configured ElevenLabs fallback never engaged, and unspoken text was appended to the context. The watchdog did not cover mid-utterance failures. Lessons:
  - Treat a reconnect that loses in-flight work as a failure.
  - Run a watchdog for audio that was expected but never arrived.
  - Commit only spoken text to the context, using word timestamps [1].

### 7.4 Guardrails, privacy and compliance

- **Guardrails.** Run classifiers concurrently with the main LLM so they do not add latency [85].
- **PII redaction.**
  - Deepgram redaction covers `pci`, `pii`, `phi` and `numbers`. It applies to transcripts only, not audio. Streaming Nova supports it but trades accuracy against `no_delay`. Flux can only redact numbers. Entity redaction is English-only [68].
  - AssemblyAI redacts **final turns only** and suppresses partial turns by default [69].
  - Recordings, traces and logs therefore need their own retention and redaction policies.
- **Data residency and on-prem.** OpenAI offers an EU SIP endpoint [20]. LiveKit trunks can be region-restricted [26]. sherpa-onnx runs fully offline [84].

### 7.5 Accessibility and multilingual

- **Language coverage.** Smart Turn v3 covers 23 languages with accuracy from 81.27% (Vietnamese) to 97.10% (Turkish) [45]; v3.1 raised English to 94.7–95.6% [46]. Flux launched English-only [49]. For languages its turn model does not cover, LiveKit recommends VAD-only detection [94].
- **Patient endpointing.** Dynamic endpointing adapts the silence window; LiveKit's example uses `min_delay` 0.3 s and `max_delay` 2.5 s [95]. Vapi falls back to `onNoPunctuationSeconds` 1.5 s [93]. Expose these per deployment for slower or disfluent speakers.
- **Interruptions.** Adaptive interruption handling separates backchannels from real interruptions, and the agent can resume after a false interruption [94][95].
- **Noise.** Krisp VIVA and ai-coustics QUAIL models reduce noise and competing voices [37].
- **Live captions.** GPT-Live streams transcript deltas, but warns that fragments are not completed turns [25].

---

## 8. On-device and edge

| Stage | Option | Evidence |
|---|---|---|
| VAD | Silero | About 2 MB; under 1 ms per 30+ ms chunk on one CPU thread; MIT license [83] |
| Turn detection | Smart Turn v3/v3.1 | 8 MB int8 ONNX; 12.6–94.8 ms on CPU [45]; 1–5 ms on GPU [46] |
| STT | whisper.cpp | Metal, Core ML encoder (about 3× faster), CUDA, Vulkan, OpenVINO; `whisper-stream` [80] |
| STT | Moonshine v2 | Sliding-window streaming encoder for the edge [79] |
| STT | sherpa-onnx | Offline streaming ASR, TTS and VAD; Linux/macOS/Windows/Android/iOS/WASM [84] |
| TTS | Kokoro-82M ONNX | About 300 MB, 80 MB quantized; "near real-time" on an M1 [82] |
| Apple Silicon | mlx-audio | Parakeet, Whisper, Kokoro, Qwen3-TTS and speech-to-speech models [78] |
| Runtimes | ONNX Runtime | CUDA, TensorRT, OpenVINO, DirectML, CoreML, QNN, XNNPACK, WebGPU execution providers [81] |
| Full local pipeline | Nemotron stack | RTX 5090: 508 ms V2V P50; DGX Spark: 1,180 ms [75] |

---

## 9. Implications for voice-agent-next

### 9.1 Transport layer

1. **A single `Transport` interface.**
   - Audio in both directions as PCM16 mono frames stamped with audio-stream time.
   - Events: `clear`, `mark` (playout acknowledgement), `dtmf`, `transfer`, `hangup`, and data messages (transcripts, metrics).
   - A capability map, e.g. `supports_clear`, `supports_mark`, `native_rates`, `dtmf`.
2. **Built-in transports, in priority order.**
   1. `local`: sounddevice.
   2. `webrtc-p2p`: aiortc, with bundled signaling, TURN configuration and relay-only mode [12][39].
   3. `websocket`: binary frames with a JSON control channel, following Vonage's design [17]. JSON with base64 is only a compatibility mode.
   4. Telephony serializers for Twilio, Telnyx, Vonage and Plivo (Table 4).
   5. Room adapters for LiveKit and Daily.
   6. SIP through LiveKit or OpenAI [20][26]. We should not write our own SIP stack at first.
3. **Audio formats.**
   - Canonical internal format: 16 kHz into STT and 24/48 kHz out of TTS.
   - Resample exactly once, at the edges, and pass telephony μ-law through when the provider accepts it [2].
   - Internal frame size is 10 ms (the APM requirement [115]); on-wire frames are 20 ms.
4. **Playout cursor per utterance.**
   - Sources, in order of preference: Twilio `mark` / Plivo checkpoint when available [13][19]; otherwise an estimate from the wall clock and buffer depth.
   - This drives interruption truncation and "spoken-only" context commits [1][63].
5. **Speech-to-speech modes.** Support *relay* (client ↔ us ↔ provider WebSocket) and *direct* (client ↔ provider WebRTC via ephemeral key, with our sideband control channel) [22][23].

### 9.2 Metrics to record

All timestamps use a monotonic clock, per turn, and each carries its audio-stream offset.

| Metric | Definition |
|---|---|
| `t0 user_speech_end` | Time of the last voiced frame = VAD-stop time − VAD hangover (`stop_secs`), as in [7] |
| `endpointing_delay` | Turn committed − t0 |
| `stt_final_latency` | Final transcript received − t0 (TTFS in [91]) |
| `llm_ttft` | First token − LLM request sent |
| `llm_ttfs` | First speakable clause released to TTS − LLM request sent [53] |
| `tts_ttfb` | First audio chunk − first text sent, **per utterance** [64] |
| `server_v2v` (platform turn gap) | First agent frame handed to the transport − t0 [2] |
| `client_v2v` (mouth-to-ear) | First agent sample rendered by the client − the client's own end-of-speech time. Reported by the client SDK, or measured by acoustic loopback in benchmarks [1][2] |
| `transport_overhead` | `client_v2v − server_v2v` |
| `greeting_latency` | Session connected → first agent audio heard [7] |
| `bargein_stop_latency` | User speech onset → agent audio stopped at the client (or `clear` sent) |
| `false_interruption_rate`, `premature_eot_rate` | Share of interruptions later judged false; share of turns where the user resumed within N ms after the agent started speaking |
| `speculation_waste` | Cancelled speculative LLM calls and tokens per turn [48] |
| `tool_latency`, `filler_played` | Duration of each tool call; whether a filler was triggered |
| Audio health | Input overflows, output underflows, jitter-buffer depth, packet loss and RTT (WebRTC statistics [1]) |
| Cost | Audio seconds in/out, LLM input/cached/output tokens [5], TTS characters, $/min |

Report P50/P90/P95/P99 tagged by turn index, transport, provider and region. Export them as OTel spans compatible with Pipecat's names where possible [6].

### 9.3 Latency optimizations built into the runtime

- **Warm-up.**
  - Pre-open STT/TTS/LLM connections and resolve DNS before the greeting [85].
  - Load VAD and turn models into the prewarmed process pool [4][40].
  - Pre-synthesize the greeting and filler clips.
- **Turn detection.** Pluggable pipeline: VAD (0.2 s) → semantic model (Smart Turn) → fallback timeout (3 s) [47], plus provider end-of-turn modes (Flux, Realtime) [94].
- **Speculative generation.** Off by default, with a cost budget. Start on eager end-of-turn, cancel on resume, and reuse the result if the normalized final transcript matches [48].
- **Clause aggregator.** Short first chunk, then sentence-sized chunks, with explicit flushes; set provider buffer limits such as Cartesia's `max_buffer_delay_ms` low [52].
- **Prompt-layout helper.** Keep the static prefix first and dynamic state last, and record cache-hit tokens [54][56].
- **Tool-call watchdog.** After N ms, play a filler or thinking sound [1][97]. Optionally prefetch predictable tools and hedge LLM calls [85].
- **Region pinning and startup latency probe.** Warn when providers sit in a different region from the worker [67].

### 9.4 Fallback and failover design

- **`ServiceChain` per STT/LLM/TTS.**
  - Health states with a circuit breaker and background probes, as in [61][62].
  - Timeouts: connect timeout, and first-result timeouts (STT partial, LLM first token, TTS first audio). Silent reconnects count as errors [63].
- **Mid-stream policy.**
  - STT: replay a 2–5 s ring buffer of audio into the backup.
  - LLM: switch before the first token; after it, only when configured to (cf. `retry_on_chunk_sent` [61]).
  - TTS: switch only if nothing has been heard yet; otherwise re-synthesize the unspoken remainder.
- **Context integrity.** Commit assistant text up to the playout cursor only [1][63].
- **Degraded mode.**
  - Cached audio: "I'm having trouble, one moment".
  - For telephony, keep the call alive (Plivo `keepCallAlive` [100]; TwiML resumes after the WebSocket closes [99]) and route to a fallback flow.

### 9.5 Cross-platform audio strategy

- **Backends.** Default to sounddevice with `blocksize=0`, `latency='low'` and a lock-free ring buffer bridged to asyncio [27]. Offer miniaudio as an optional backend [31]. Do not use PyAudio.
- **Linux doctor command.** Detect PortAudio 19.6 builds that lack a Pulse host API and advise using the ALSA `default`/`pipewire` devices [30].
- **Device selection.** By name substring or ID, with hot-plug re-open, and warnings when a device runs at an unexpected sample rate.
- **Platform options.** Opt-in WASAPI exclusive mode and the Core Audio `change_device_parameters` flag [28][33].
- **Echo control.** An `EchoCanceller` interface with implementations:
  1. WebRTC APM through livekit.rtc, with the playback stream fed as reference, 10 ms frames and a measured stream delay [36][115].
  2. OS-level AEC (PipeWire echo-cancel [34]; macOS and Windows voice processing (unverified)).
  3. **Headphone mode**.
  4. **Half-duplex** (mic gated while the agent speaks, no barge-in) as the safe fallback [113].

### 9.6 Deployment story

- **`voice-agent-next dev`.** A single process with local audio or WebRTC P2P and a bundled test page.
- **`voice-agent-next serve`.**
  - A supervisor runs a prewarmed worker pool with one session per process by default, so CPU-heavy VAD and resampling in one session cannot starve another.
  - CPU-load admission control (default 0.7 [40]) with a `/load` endpoint for autoscalers.
  - Graceful drain and OTel/Prometheus exporters.
- **Reference Compose stacks.** CPU orchestrator plus GPU vLLM/STT/TTS services, following NVIDIA's blueprint and Unmute [72][76]. Document VRAM needs and per-GPU stream density [73][74].
- **Benchmark suite.**
  - Acoustic-loopback V2V across local, WebRTC, WebSocket and a Twilio PSTN path [1][85].
  - Concurrency sweeps to capture TTFT degradation under load [77].
  - Per-minute cost computed from recorded usage.

---

## 10. Sources

1. Voice AI & Voice Agents: An Illustrated Primer. https://voiceaiandvoiceagents.com/ (Feb 2025, updated Jun 2026)
2. Twilio, "Core Latency in AI Voice Agents" (P. Bredeson). https://www.twilio.com/en-us/blog/developers/best-practices/guide-core-latency-ai-voice-agents (2025-11-17)
3. WebRTC.ventures, "The Voice AI Latency Budget: Where Every Millisecond Goes". https://webrtc.ventures/2026/09/voice-ai-latency-budget/ (2026-09-23)
4. LiveKit, "Understand and Improve Voice Agent Latency". https://livekit.com/blog/understand-and-improve-agent-latency (2026-04-13)
5. LiveKit Docs, "Capturing metrics". https://docs.livekit.io/agents/ops/logging/ (accessed 2026-09-24)
6. Pipecat Docs, "OpenTelemetry Tracing". https://docs.pipecat.ai/server/utilities/opentelemetry (accessed 2026-09-24)
7. Pipecat Docs, "User-Bot Latency Observer". https://docs.pipecat.ai/server/utilities/observers/user-bot-latency-observer (accessed 2026-09-24)
8. Modal, "One-second voice-to-voice latency with Modal, Pipecat, and open models". https://modal.com/blog/low-latency-voice-bot (2025-11-04)
9. LiveKit, "Why WebRTC beats WebSockets for realtime voice AI". https://livekit.com/blog/why-webrtc-beats-websockets-for-voice-ai-agents (2026-03-23)
10. BlogGeek.me, "WebRTC for Voice AI: how the transport layer works". https://bloggeek.me/voice-ai/ (2026)
11. Daily, "You don't need a WebRTC server for your voice agents". https://www.daily.co/blog/you-dont-need-a-webrtc-server-for-your-voice-agents/ (2025-07-21)
12. DEV Community (P. Santus), "Switching my AI voice agent from WebSocket to WebRTC". https://dev.to/aws-builders/switching-my-ai-voice-agent-from-websocket-to-webrtc-what-broke-and-what-i-learned-3dkn (2026-03-27)
13. Twilio Docs, "Media Streams WebSocket Messages". https://www.twilio.com/docs/voice/media-streams/websocket-messages
14. Twilio Docs, "Media Streams". https://www.twilio.com/docs/voice/media-streams
15. Telnyx, "Real-Time Media Streaming with Expanded Codec Support". https://telnyx.com/release-notes/media-streaming-codec-update (2025-09-08)
16. Telnyx Docs, TeXML `<Stream>`. https://developers.telnyx.com/docs/voice/programmable-voice/texml-verbs/stream
17. Vonage Docs, "WebSockets". https://developer.vonage.com/en/voice/voice-api/concepts/websockets
18. Plivo Docs, "Audio Streaming (XML)". https://www.plivo.com/docs/voice/xml/audio-streaming
19. Plivo, "Audio Streaming for Voice AI Agents". https://www.plivo.com/audio-streaming/
20. OpenAI Docs, "Realtime API with SIP". https://developers.openai.com/api/docs/guides/realtime-sip
21. OpenAI Docs, "Realtime API with WebSocket". https://developers.openai.com/api/docs/guides/realtime-websocket
22. OpenAI Docs, "Realtime API with WebRTC". https://developers.openai.com/api/docs/guides/realtime-webrtc
23. OpenAI Docs, "Getting started with GPT-Live". https://developers.openai.com/api/docs/guides/live
24. OpenAI Docs, "GPT-Live 1 model". https://developers.openai.com/api/docs/models/gpt-live-1
25. DataCamp, "GPT-Live-1 API Tutorial". https://www.datacamp.com/tutorial/gpt-live-1-api (2026-09-15)
26. LiveKit Docs, "Telephony introduction". https://docs.livekit.io/telephony/
27. python-sounddevice Docs, "Streams". https://python-sounddevice.readthedocs.io/en/latest/api/streams.html
28. python-sounddevice Docs, "Platform-specific settings". https://python-sounddevice.readthedocs.io/en/latest/api/platform-specific-settings.html
29. python-sounddevice Docs, "Installation". https://python-sounddevice.readthedocs.io/en/latest/installation.html
30. python-sounddevice issue #609, "no Pulse/PipeWire devices listed". https://github.com/spatialaudio/python-sounddevice/issues/609 (2025-11)
31. pyminiaudio (GitHub). https://github.com/irmen/pyminiaudio
32. PyAudio homepage. https://people.csail.mit.edu/hubert/pyaudio/ (v0.2.14, 2024-11-23)
33. Microsoft Learn, "Low Latency Audio". https://learn.microsoft.com/en-us/windows-hardware/drivers/audio/low-latency-audio (updated 2025-03-26)
34. PipeWire Docs, "Echo Cancel module". https://docs.pipewire.org/page_module_echo_cancel.html
35. PipeWire Docs, "pipewire-props(7)". https://docs.pipewire.org/page_man_pipewire-props_7.html
36. LiveKit python-sdks README. https://github.com/livekit/python-sdks/blob/main/README.md
37. LiveKit Docs, "Noise & echo cancellation". https://docs.livekit.io/transport/media/noise-cancellation/
38. aiortc (GitHub). https://github.com/aiortc/aiortc
39. Pipecat Docs, "Small WebRTC Transport". https://docs.pipecat.ai/api-reference/server/services/transport/small-webrtc
40. LiveKit Docs, "Server options". https://docs.livekit.io/agents/server/options/
41. LiveKit, "Why is my agent slow to join a room?". https://livekit.com/blog/agent-join-latency (2026-08-05)
42. Pipecat Cloud Docs, "Scaling". https://docs.pipecat.ai/pipecat-cloud/fundamentals/scaling
43. Daily, "Pipecat Cloud Pricing". https://www.daily.co/pricing/pipecat-cloud/ (accessed 2026-09-24)
44. Daily, "Pipecat Cloud is now generally available". https://www.daily.co/blog/pipecat-cloud-is-now-generally-available/ (2026-01-08)
45. Daily, "Announcing Smart Turn v3, with CPU inference in just 12ms". https://www.daily.co/blog/announcing-smart-turn-v3-with-cpu-inference-in-just-12ms/ (2025-09-11)
46. Daily, "Improved accuracy in Smart Turn v3.1". https://www.daily.co/blog/improved-accuracy-in-smart-turn-v3-1/ (2025-12-03)
47. Pipecat Docs, "Smart Turn Overview". https://docs.pipecat.ai/api-reference/server/utilities/turn-detection/smart-turn-overview
48. Deepgram Docs, "Optimize Voice Agent Latency with Eager End of Turn". https://developers.deepgram.com/docs/flux/voice-agent-eager-eot
49. Deepgram, "Introducing Flux: Conversational Speech Recognition". https://deepgram.com/learn/introducing-flux-conversational-speech-recognition (2025-10)
50. Deepgram Docs, "Measuring STT Latency". https://developers.deepgram.com/docs/measuring-streaming-latency
51. ElevenLabs Docs, "Understanding latency". https://elevenlabs.io/docs/eleven-api/concepts/latency
52. Cartesia Docs, "Text-to-Speech (WebSocket)". https://docs.cartesia.ai/api-reference/tts/websocket
53. MarkTechPost, "Lowest-Latency Inference APIs for Voice and Realtime Agents: A TTFT-First Benchmark". https://www.marktechpost.com/2026/08/30/lowest-latency-inference-apis-for-voice-and-realtime-agents-a-time-to-first-token-ttft-first-benchmark/ (2026-08-30)
54. OpenAI Docs, "Prompt caching". https://developers.openai.com/api/docs/guides/prompt-caching
55. OpenAI Cookbook, "Prompt Caching 201". https://developers.openai.com/cookbook/examples/prompt_caching_201
56. Anthropic Docs, "Prompt caching". https://platform.claude.com/docs/en/build-with-claude/prompt-caching
57. Stivers et al., "Universals and cultural variation in turn-taking in conversation", PNAS (abstract). https://research.rug.nl/en/publications/universals-and-cultural-variation-in-turn-taking-in-conversation/ (2009-06-30)
58. Levinson & Torreira, "Timing in turn-taking and its implications for processing models of language", Front. Psychol. 6:731. https://www.mpi.nl/publications/item2161027/timing-turn-taking-and-its-implications-processing-models-language (2015)
59. ITU-T Rec. G.114, "One-way transmission time". https://www.itu.int/rec/T-REC-G.114-200305-I/en (2003-05)
60. LiveKit Docs, "Testing and evaluation". https://docs.livekit.io/agents/start/testing/
61. LiveKit Docs, "Fallback strategies". https://docs.livekit.io/agents/logic/fallback-strategies/
62. Pipecat Docs, "ServiceSwitcher". https://docs.pipecat.ai/api-reference/server/utilities/service-switchers/service-switcher
63. Pipecat issue #5305, "TTS text that was never spoken is appended to the LLM context…". https://github.com/pipecat-ai/pipecat/issues/5305 (2026-08-12)
64. Pipecat issue #3451, "WebSocket TTS services only measure TTFB on first request". https://github.com/pipecat-ai/pipecat/issues/3451 (2026-01-14)
65. livekit/agents issue #3824, "How to measure latency properly?". https://github.com/livekit/agents/issues/3824 (2025-11-07)
66. livekit/agents issue #3236, "e2e latency should include on_user_turn_completed_delay". https://github.com/livekit/agents/issues/3236 (2025-08-22)
67. livekit/agents issue #4053, "Latency increase when deploying to LiveKit Cloud (EU region)". https://github.com/livekit/agents/issues/4053 (2025-11-22)
68. Deepgram Docs, "Redaction". https://developers.deepgram.com/docs/redaction
69. AssemblyAI Docs, "PII Redaction (Streaming)". https://www.assemblyai.com/docs/streaming/pii-redaction
70. open-telemetry/semantic-conventions-genai PR #394, "Realtime / live voice model generation". https://github.com/open-telemetry/semantic-conventions-genai/pull/394 (open, Sep 2026)
71. OpenTelemetry Blog, "Inside the LLM Call: GenAI Observability with OpenTelemetry". https://opentelemetry.io/blog/2026/genai-observability/ (2026)
72. kyutai-labs/unmute (GitHub). https://github.com/kyutai-labs/unmute
73. Kyutai, "Kyutai STT". https://kyutai.org/stt/
74. NVIDIA on Hugging Face, "Scaling Real-Time Voice Agents with Cache-Aware Streaming ASR". https://huggingface.co/blog/nvidia/nemotron-speech-asr-scaling-voice-agents (2026-01-05)
75. Daily, "Building Voice Agents with NVIDIA Open Models". https://www.daily.co/blog/building-voice-agents-with-nvidia-open-models/ (2026-01-05)
76. NVIDIA-AI-Blueprints/nemotron-voice-agent (GitHub). https://github.com/NVIDIA-AI-Blueprints/nemotron-voice-agent
77. BentoML, "Benchmarking LLM Inference Backends". https://www.bentoml.com/blog/benchmarking-llm-inference-backends (2024-06-05)
78. Blaizzy/mlx-audio (GitHub). https://github.com/Blaizzy/mlx-audio
79. "Moonshine v2: Ergodic Streaming Encoder ASR…", arXiv 2602.12241. https://arxiv.org/abs/2602.12241 (2026-02-12)
80. ggml-org/whisper.cpp (GitHub). https://github.com/ggml-org/whisper.cpp
81. ONNX Runtime, "Execution Providers". https://onnxruntime.ai/docs/execution-providers/
82. thewh1teagle/kokoro-onnx (GitHub). https://github.com/thewh1teagle/kokoro-onnx
83. snakers4/silero-vad (GitHub). https://github.com/snakers4/silero-vad
84. sherpa-onnx documentation. https://k2-fsa.github.io/sherpa/onnx/index.html
85. Cresta, "Engineering for Real-Time Voice Agent Latency" (D. Hoske). https://cresta.com/blog/engineering-for-real-time-voice-agent-latency (2025-10-21)
86. SignalWire, "Voice AI Providers Are Lying to You About Latency". https://signalwire.com/blogs/industry/ai-providers-lying-about-latency (2025-10-08)
87. Telnyx, "Voice AI agents compared on latency". https://telnyx.com/resources/voice-ai-agents-compared-latency (2026)
88. Coval, "Voice AI Latency: What Causes Delays and How to Fix Them". https://www.coval.ai/blog/voice-ai-latency (2026-03-03)
89. DestiLabs, "2026 AI Voice Agent Benchmark: Latency & Cost per Minute". https://www.destilabs.com/blog/ai-voice-agent-benchmark-2026 (2026-06-30)
90. Deepgram, "Voice AI Deployment Cost: Cloud vs Dedicated vs Self-Hosted". https://deepgram.com/learn/voice-ai-deployment-cost-cloud-dedicated-self-hosted
91. pipecat-ai/stt-benchmark (GitHub). https://github.com/pipecat-ai/stt-benchmark
92. Hamming, "Voice Agent Testing Guide". https://hamming.ai/resources/voice-agent-testing-guide (2026-01-23)
93. Vapi Docs, "Speech configuration". https://docs.vapi.ai/customization/speech-configuration
94. LiveKit Docs, "Turns overview". https://docs.livekit.io/agents/logic/turns/
95. LiveKit, "Configuring turn detection and interruptions in LiveKit Agents". https://livekit.com/blog/turn-detection-and-interruption-handling (2026-06-30)
96. LiveKit, "Turn detection for voice agents: VAD, endpointing, and model-based detection". https://livekit.com/blog/turn-detection-voice-agents-vad-endpointing-model-based-detection (2026-02-21)
97. LiveKit Docs, "Background audio". https://docs.livekit.io/agents/multimodality/audio/background-audio/
98. Pipecat Docs, "AudioBufferProcessor". https://docs.pipecat.ai/server/utilities/audio/audio-buffer-processor
99. Pipecat Docs, "Twilio WebSocket Integration". https://docs.pipecat.ai/pipecat/telephony/twilio-websockets
100. Pipecat Docs, "Plivo WebSocket Integration". https://docs.pipecat.ai/pipecat/telephony/plivo-websockets
101. pipecat-ai/pipecat (GitHub README). https://github.com/pipecat-ai/pipecat
102. livekit/agents (GitHub README). https://github.com/livekit/agents
103. PyPI, livekit 1.1.20. https://pypi.org/project/livekit/ (2026-09-23)
104. PyPI, daily-python 0.32.0. https://pypi.org/project/daily-python/ (2026-08-19)
105. PyPI, sounddevice 0.5.6. https://pypi.org/project/sounddevice/ (2026-08-17)
106. PyPI, pipecat-ai 1.11.0. https://pypi.org/project/pipecat-ai/ (2026-09-18)
107. PyPI, livekit-agents 1.8.3. https://pypi.org/project/livekit-agents/ (2026-09-23)
108. PyPI, aiortc 1.15.0. https://pypi.org/project/aiortc/ (2026-07-13)
109. Opus Codec homepage. https://opus-codec.org/
110. MDN, "MediaTrackSettings: echoCancellation". https://developer.mozilla.org/en-US/docs/Web/API/MediaTrackSettings/echoCancellation
111. MDN, "AudioWorklet". https://developer.mozilla.org/en-US/docs/Web/API/AudioWorklet
112. FutureAGI, "How to Optimize LiveKit Voice Agent Latency in 2026". https://futureagi.com/blog/how-to-optimize-livekit-latency-2026/ (2026-03-06, updated 2026-05-20)
113. Pipecat issue #188, "Interrupted by itself when speaker on". https://github.com/pipecat-ai/pipecat/issues/188
114. OpenAI Docs, "Audio and voice". https://developers.openai.com/api/docs/guides/audio
115. livekit/python-sdks source, `livekit-rtc/livekit/rtc/apm.py` (AudioProcessingModule docstrings). https://raw.githubusercontent.com/livekit/python-sdks/main/livekit-rtc/livekit/rtc/apm.py
