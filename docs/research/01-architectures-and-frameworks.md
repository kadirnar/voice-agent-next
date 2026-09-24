# Voice-Agent Architectures & Open-Source Frameworks (state of the art, September 2026)

> Research note 01 for **voice-agent-next**, compiled 2026-09-24.
> **Method.** I read primary sources directly: official docs, READMEs, release notes, source files and papers. Repository metrics (stars, licenses, releases, CI matrices) come from the GitHub API, and release and download data from PyPI and pypistats, all on 2026-09-24. Latency and accuracy figures from vendors are their own claims. Anything I could not confirm from a primary source is marked *(unverified)*. Citations `[n]` point to §7.

---

## 1. Executive summary

- **Cascaded is still the default for production, but it's no longer the only serious option.** LiveKit's docs call STT-LLM-TTS "the right default" for most production agents [34]. The June 2026 update of the Voice AI primer says speech-to-speech (S2S) models "do not follow instructions or call tools as reliably as text-mode LLMs". It also says that in practice OpenAI's and Google's S2S models are "slower than a well-tuned cascaded" agent [1]. Streaming overlap cuts a naive cascade from 1–2 s+ down to about 400–800 ms [49].
- **The big shift in 2026 is full-duplex plus delegation (a "talker/thinker" split).** OpenAI released `gpt-live-1` in the API on 2026-09-10 [7][8]. It listens while it speaks, decides for itself when to yield to the user, and hands reasoning and tools to a backend you choose (OpenAI Responses or your own agent). It costs $0.05/min [5][6]. Research got here first: full-duplex models (Moshi, PersonaPlex [64][66]) and "tandem" designs that pair an S2S model with a backend LLM (KAME [67]). Frameworks already had delegation-like patterns too (Pipecat's async function calls, LiveKit's "subagent delegation" [15][43]). LiveKit shipped support the same day (1.8.1, which adds a new `DuplexModel` class), and Pipecat followed the next day (1.9.0) [33][16]. But the model breaks turn-based assumptions. Barge-in is model-controlled, context is append-only, there's no exact-script `say()`, and there's no half-cascade mode [36][16].
- **Turn-taking is now a layered ML subsystem, not a silence timer.** The layers are VAD, then an audio end-of-turn model, optionally LLM completion markers, then a backchannel/interruption classifier, preemptive generation and false-interruption recovery [1][21][38][39][40]. Permissively licensed models exist: Smart Turn v3.2 is BSD-2, about 8M params, 8 MB as int8, and covers 23 languages [29]. LiveKit's best models, by contrast, are either served from its cloud or licensed only for use inside LiveKit Agents [38][47], and self-hosting users have filed feature requests for local alternatives [51].
- **Two frameworks dominate open source.**
  - **Pipecat:** BSD-2, 15.8k★, reached 1.0 on 2026-04-14 and is now at 1.11, about 1.0M PyPI downloads last month.
  - **LiveKit Agents:** Apache-2.0, 14.3k★, reached 1.0 on 2025-04-10 and is now at 1.8.2, about 3.2M downloads last month [14][33][52][53].

  Pipecat is a frame-processing pipeline plus a worker bus. LiveKit is an `AgentSession`/`Agent` pair with overridable "nodes", built on WebRTC rooms.
- **The OpenAI Realtime event protocol is turning into a de facto wire standard.** Hugging Face speech-to-speech v1.0 (2026-09-06) implements its core event set over WebSocket and WebRTC and is CI-tested against the OpenAI Agents SDK. Kyutai Unmute's browser protocol is also "based on" it [60][62].
- **Local, cross-platform use is the weakest area.**
  - **Unmute:** Linux or WSL only, x86_64, and a CUDA GPU with 16 GB+ [62].
  - **HF s2s:** CI runs on Ubuntu and macOS only, and the README recommends headphones to avoid echo [60].
  - **Pipecat:** tests run on Ubuntu only, and its Daily transport ships no Windows wheels [28][52].
  - **TEN:** Python 3.10 only [56].

  Of the frameworks whose CI I checked (Pipecat, LiveKit, HF s2s), only LiveKit tests on Linux, macOS and Windows. It's also the only one with built-in echo cancellation in its local console mode [46].
- **Watch the licenses.** TEN Framework, TEN VAD and TEN turn detection add non-OSI restrictions on top of Apache-2.0. All three forbid deployments that compete with Agora, and the framework and turn-detection licenses also forbid hosting on end-user devices [55]. LiveKit's text turn-detector weights may only be used with LiveKit Agents [47].
- **Several projects have stalled or been merged away.**
  - **Vocode:** last commit 2024-11-15 [58].
  - **FastRTC:** last release 2025-11-24 [61].
  - **RealtimeVoiceChat:** the author stepped back from maintaining it [84].
  - **Pipecat side-repos:** `pipecat-flows` and `pipecat-subagents` were folded into Pipecat core and archived [24][32].
- **New since 2025:** GetStream Vision Agents, Dograh (an open-source Vapi/Retell alternative built on a Pipecat fork), Cartesia Line, AWS Strands `BidiAgent`, Google ADK live agents, HF speech-to-speech 1.0 and NVIDIA PersonaPlex [69][70][71][72][73][60][66].
- **Testing is moving into the frameworks.**
  - **Pipecat Evals:** scripted scenarios plus LLM-simulated callers, with a text-only mode for speed [25].
  - **LiveKit:** a built-in test framework with LLM judges [33].
  - **Open benchmarks:** eot-bench for end-of-turn detection [50] and stt-benchmark for STT latency [30].

  No framework benchmarks the *architectures* against each other end to end. That's an opening for voice-agent-next.

---

## 2. Architecture patterns

### 2.1 The four patterns

| | **Cascaded** | **Half-cascade** | **Realtime S2S (turn-based)** | **Full-duplex S2S (+ delegation)** |
|---|---|---|---|---|
| Data path | audio → VAD/turn detector → STT → text LLM → TTS → audio | audio → audio-input LLM → text → TTS | audio ↔ one speech model; server VAD takes turns | continuous audio in/out; model decides when to speak; reasoning/tools delegated |
| 2026 examples | Pipecat/LiveKit pipelines, HF s2s, Unmute, Vapi, Retell | Ultravox (audio in, streaming text out) [65]; realtime model in text modality + TTS [35]; HF s2s `--stt none` + audio-input chat model [60] | OpenAI Realtime (`gpt-realtime-2.1` [11]), Gemini Live, Nova Sonic, Grok Voice Agent API [35] | OpenAI GPT-Live-1 [5]; Moshi [64]; NVIDIA PersonaPlex [66] |
| Latency | ~400–800 ms streaming vs 1–2 s+ blocking [49]; 500 ms shown with co-located GPUs [1] | "Moderate" [34] | "Fastest" per LiveKit [34]; the primer says lower latency is possible "in theory", but OpenAI's and Google's are slower than a tuned cascade in practice [1] | Moshi 160 ms theoretical / 200 ms practical [64]; GPT-Live turn-taking 0.798 s vs 1.41 s for GPT-Realtime-2.1 (OpenAI via [8]) |
| Exact scripted speech | Yes (`say()`) [34] | Yes [34] | No [34][35] | No; persona fixed per session [36] |
| Tool calling | "Mature" [34] | "Less mature" [34] | "Less mature" [34] | Via backend delegation (GPT-Live) [5]; none (PersonaPlex) [37] |
| Transcripts / audit | Full text trail, interim transcripts [34] | Output text only [34] | User transcripts delayed [35] | Items arrive after the audio [35]; no user transcript (PersonaPlex) [37] |
| Understanding | Loses prosody [34] | Prosody-aware [34] | Prosody-aware; far better on mixed-language speech [1] | Native overlap, backchannels, barge-in [35] |
| Cost | Lowest: a Realtime-API agent was 3–5× a GPT-4.1 cascade [1] | Mid | High [1] | $0.05/min front-end + backend tokens [6] |

**Observations**

1. **Frameworks now name all four patterns explicitly.** LiveKit documents pipeline, realtime and half-cascade side by side [34]. OpenAI's guide compares GPT-Live, the Realtime API and chained pipelines, and advises: "choose the audio architecture first, then design the rest of the agent workflow the same way you would for text" [4].
2. **Full duplex breaks turn-shaped abstractions.** On GPT-Live, LiveKit's docs say: "the framework can stop the audio it plays to the user, but the model keeps talking until it stops on its own" [36]. Output streams continuously, silence included, so the framework has to derive turn boundaries from the audio [35]. Context is append-only, and instructions can't change mid-session [36].
3. **The talker/thinker split is converging across stacks.** Examples:
   - **KAME (2025):** relays the query to a backend LLM and injects its text into the S2S model's speech in real time, getting close to cascade correctness at S2S latency [67].
   - **GPT-Live:** productizes the split with "client" or "Responses" delegation [5].
   - **LiveKit:** a "subagent delegation" pattern [43].
   - **Pipecat 1.0:** async tools (`cancel_on_interruption=False`) that stream intermediate results [15].
   - **Retell:** its custom-LLM WebSocket makes the platform the talker and your server the thinker [76].
4. **Duplex needs its own evaluations.** Full-Duplex-Bench scores pause handling, backchanneling, turn-taking and interruptions [68]. The September 2026 ECHO benchmark finds that most duplex systems are biased toward yielding the floor [68].

### 2.2 Latency budgets

The primer's worked example is a macOS client talking to a cloud agent [1]:

| Stage | ms |
|---|---|
| Inbound audio: mic 40, Opus encode 21, network 10, packet handling 2, jitter buffer 40, decode 1 | 114 |
| Transcription + endpointing | 300 |
| LLM time-to-first-byte | 650 |
| Sentence aggregation | 20 |
| TTS time-to-first-byte | 120 |
| Outbound audio: encode 21, packets 2, network 10, jitter 40, decode 1, speaker 15 | 89 |
| **Voice-to-voice total** | **1,293** |

Endpointing plus LLM TTFB account for about 73% of the total.

**How targets have shifted**

- **2024:** "800 ms voice-to-voice latency is a good target" [2].
- **2026:** the primer now says "people are happy talking to agents that respond within 1,500 ms", but warns against latency spikes as features like tool calling are added [1].
- **Vendor figures:** Retell says its proprietary turn-taking runs at about 600 ms [75]. Vapi's docs walk through roughly 2.3 s total with default settings, ranging from 1.9 s (aggressive) to 4.7 s (conservative) [74].

**Measuring latency**

- The primer notes that most tools only measure time-to-first-byte proxies. It recommends measuring from recordings, end of user speech to start of bot speech [1].
- OpenAI recommends tracking the median and p95, and timestamping every observable stage: delegation receipt, backend start, tool start/end, audio arrival and playback [4].
- Pipecat's stt-benchmark argues that "tail latency matters more than median" [30].

### 2.3 The turn-taking stack

| Layer | Representative implementations | Where it runs / license |
|---|---|---|
| VAD | Silero (<1 ms per 30 ms chunk, one CPU thread [21]); Krisp VIVA; TEN VAD | Local CPU. Silero: MIT. Krisp: commercial. TEN VAD: Apache-2.0 + Agora conditions [55] |
| Audio end-of-turn model | **Smart Turn v3.2**: Whisper-Tiny backbone + linear head, ~8M params, 8 MB int8 / 32 MB fp32, 23 languages, 16 kHz input (≤8 s), ~10 ms on some CPUs and <100 ms on most cloud instances [29]. **LiveKit TurnDetector v1** (cloud) / **v1-mini** (local CPU), 14 languages [38][48] | Smart Turn: BSD-2, local. LiveKit v1 via LiveKit Inference; v1-mini bundled in the SDK |
| Text end-of-turn model | LiveKit text detector: Qwen2.5-0.5B-derived, 135M params, <500 MB RAM, deprecated ahead of 2.0 [38][47]; TEN turn detection [55] | LiveKit Model License: use "only ... with the LiveKit Agents framework" [47]; TEN: Agora conditions |
| Turn events from the STT | Deepgram Flux: `StartOfTurn` / `EagerEndOfTurn` / `TurnResumed` / `EndOfTurn`, ~260 ms EoT claimed [78]; Kyutai STT-1B semantic VAD (0.5 s delay) [63]; others via "external turn management" [22] | Vendor cloud (Kyutai is self-hosted) |
| LLM-in-the-loop markers | Pipecat mixin: the LLM's first token is ● (complete), ◐ (short wait) or ○ (long wait). An incomplete marker suppresses speech and arms a re-prompt timeout [28]. The primer calls VAD + audio model + LLM marker the state of the art [1] | BSD-2 |
| Provider-side | OpenAI `server_vad` (default) / `semantic_vad` with `interrupt_response` [9]; full-duplex models own their turns [35] | Provider |
| Barge-in classification | LiveKit adaptive interruption (cloud; drops backchannels) [39]; Krisp VIVA interruption prediction and min-words strategies [22]; Vapi `stopSpeakingPlan` (`voiceSeconds` 0.2, `backoffSeconds` 1.0) [74] | Mixed |
| Speculation & recovery | LiveKit preemptive generation: on by default, TTS opt-in, skipped for utterances >10 s, max 3 retries. False-interruption resume after 2.0 s [40]. Flux `EagerEndOfTurn` → `TurnResumed` [78] | — |

**Why the choice matters**

- On LiveKit's own eot-bench, at a 300 ms budget, false-cutoff rates are 9.9% for v1, 12.9% for Deepgram Flux and 27.7% for ultraVAD (vendor-run) [48].
- LiveKit's docs show a text-based model committing a turn three times on mid-sentence pauses, where the audio model correctly waits [38].
- Sample-rate contracts are fragile: Smart Turn silently degraded on 8 kHz telephony audio (Pipecat #3844) [31].

### 2.4 How frameworks handle interruptions

| Approach | Where | Mechanism | Keeping context in sync with what was heard |
|---|---|---|---|
| Broadcast and flush | Pipecat | `InterruptionFrame` is a high-priority SystemFrame sent both upstream and downstream. Processors cancel their work and drop queued frames; the transport drains unplayed audio [22] | The assistant aggregator commits only the `TTSTextFrame`s that were actually played [22] |
| Speech handles | LiveKit | `say()` / `generate_reply()` return a `SpeechHandle` with `allow_interruptions`; `session.interrupt()` stops the agent manually; the agent can resume after a false interruption [42][40] | History is truncated to "only the portion of the speech that the user heard" [40] |
| Response IDs | Retell custom LLM | Each `response_required` event carries a `response_id`; responses with an outdated ID are discarded [76] | Platform-managed |
| Item truncation | OpenAI Realtime | `response.cancel` + `conversation.item.truncate` (`audio_end_ms`). Automatic over WebRTC/SIP, manual over WebSocket [10] | Server truncates the audio, but the model "doesn't have enough information to precisely align transcript and audio" [10] |
| Model-owned | GPT-Live, duplex | The model decides when to stop; the framework can only stop playback [35] | Append-only context [36] |

### 2.5 Transports and wire protocols

- **WebRTC vs WebSockets.** Use WebRTC for client↔server: no TCP head-of-line blocking, plus browser echo cancellation and noise suppression. WebSockets are fine for server↔server and prototypes [1][2]. GPT-Live offers WebRTC, WebSocket and SIP, plus a server-side "sideband" control socket [5].
- **Telephony** means media streams over WebSocket with a serializer. Pipecat has serializers for Twilio, Telnyx, Plivo, Exotel, Genesys and Vonage [14]. LiveKit has native SIP [33].
- **Client↔agent protocols.**
  - **RTVI 1.0:** Pipecat's protocol, released June 2025 [26].
  - **OpenAI Realtime events:** HF s2s calls its implementation "a tested core subset, not a claim of full ... equivalence" [60]. Unmute uses "ORA" with its own extensions [62].
  - **Media over QUIC (MoQ):** Pipecat has a MoQ transport [23].
- **Local mic/speaker.**
  - **Pipecat:** `LocalAudioTransport` uses PyAudio with no echo cancellation in its source [28].
  - **HF s2s:** recommends headphones, or `--local_audio_block_mic_during_playback`, which also disables barge-in [60].
  - **LiveKit:** console mode runs WebRTC APM (echo cancellation, noise suppression, high-pass filter, AGC) and feeds it a reverse (render) stream [46].

---

## 3. Framework profiles

### 3.1 Pipecat (Daily + community)

- **Core abstractions.**
  - **Frames:** typed dataclasses flowing through `FrameProcessor`s (`process_frame` / `push_frame`), linked in a `Pipeline` or `ParallelPipeline`.
  - **Two lanes:** SystemFrames (input audio, interruptions, errors) are high priority. DataFrames and ControlFrames are processed in strict order [17][18].
  - **Workers (1.x):** a `PipelineWorker` runs a pipeline. `WorkerRunner` manages workers over a bus (`AsyncQueueBus`, `RedisBus`, `PgmqBus`) for handoffs, parallel jobs and distributed agents [19][20].
  - **Context:** lives in an `LLMContext`, with user/assistant aggregators placed around the LLM and TTS.
- **Turn detection.** Composable start/stop "user turn strategies". The defaults are Silero VAD (`stop_secs=0.2`) and `LocalSmartTurnAnalyzerV3`. Alternatives include a speech timeout, min-words, provider-driven turns and Krisp interruption prediction [21][22]. It also offers the LLM marker mixin [28].
- **Interruptions.** As described in §2.4, with mute strategies to protect bot speech [22].
- **Providers.** The README table lists 23 STT, 25 LLM, 31 TTS and 5 S2S providers [14]. `OpenAILiveLLMService` (GPT-Live) arrived in v1.9.0 (2026-09-11) and Gemini 3.8 Live in v1.11.0 [16]. Local options include Whisper, Moonshine, FunASR/SenseVoice, Ollama, Kokoro, Piper, Pocket TTS and XTTS [27].
- **Transports.** Daily, LiveKit, SmallWebRTC (P2P), FastAPI WebSocket, WebSocket server, WhatsApp, Vonage, MoQ, avatar transports, local PyAudio/Tk, and telephony serializers [14][23][28]. Client SDKs are JS, React, React Native, iOS, Android and C++, and speak the RTVI protocol [27][26].
- **Flows.** Node-graph conversations: `FlowConfig`, node/edge functions, actions and context strategies. Shipped in core as `pipecat.flows` since 1.5.0 [24].
- **Platform and tooling.** Python ≥3.11. Tests run on Ubuntu only [28]. `daily-python` ships only Linux and macOS wheels [52]. Tooling includes the Whisker debugger, the Tail terminal dashboard, a CLI (scaffold, evals, deploy) and Pipecat Cloud/Enterprise [14][27].
- **Strengths.** Vendor-neutral with the broadest catalog of integrations. Very composable. New models get adopted within days. Explicit multi-agent bus. Behavioral evals that the project runs before every release against 100+ example agents [25].
- **Pain points.**
  - Ordering and push discipline are the developer's job ("Order matters"; processors must push every frame) [18].
  - Shared mutable context raced with queued frame updates, so 1.0 added `LLMMessagesTransformFrame` to fix it [15].
  - 1.0 removed many deprecated APIs [15].
  - Issues on record [31]: a memory leak on Ubuntu (#3116), freezes with thread-pool concurrency (#1912), and missing `ErrorFrame`s on init failures (#2876). Still open: multi-participant desync on the LiveKit transport (#3218), a shutdown race (#3757), and a request to decouple how async tool calls are rendered into context from `cancel_on_interruption` (#4657). §5 covers more.
  - Third-party guides advise running one session per process [83].

### 3.2 LiveKit Agents

- **Core abstractions.**
  - **Server and session:** an `AgentServer` handles dispatch and runs each job in its own process, with prewarmed idle processes [44]. It calls `entrypoint(JobContext)`, which creates an `AgentSession(stt, llm, tts, vad, turn_handling)` hosting one active `Agent` (instructions, tools, `chat_ctx`) [33][42].
  - **Nodes and hooks:** behavior is customized by overriding nodes (`stt_node`, `llm_node`, `tts_node`, `transcription_node`, `realtime_audio_output_node`) or hooks (`on_enter`, `on_exit`, `on_user_turn_completed`) [41].
  - **Workflows:** agents, handoffs, tasks with typed results, task groups, supervisor, and subagent delegation [43].
- **Pipeline types.** Pipeline, realtime and half-cascade are all first class; a realtime model is simply passed as `llm=` [34][35]. Release 1.8.1 (2026-09-10) added a `DuplexModel` class for models that "speak and listen simultaneously", with GPT-Live as the first implementation. An adapter presents it to the session as a realtime model [33][36].
- **Turn detection.** The default is the audio `TurnDetector`: v1 in the cloud, with a sticky fallback to local v1-mini. With the detector, endpointing runs 0.3–2.5 s, in fixed or dynamic mode. The text detector is deprecated [38].
- **Interruptions.** "Adaptive" mode runs in the cloud; otherwise VAD. Also `min_words`, `min_duration`, false-interruption resume, and preemptive generation [39][40].
- **Providers.** About 75 plugin packages, including 9 realtime providers (among them GPT-Live, Gemini Live, Nova Sonic, Ultravox and PersonaPlex) [33][35]. "Provider/model" strings via LiveKit Inference [33]. Local options are thin: Ollama, Kokoro through an OpenAI-compatible server, NVIDIA Riva, and self-hosted PersonaPlex [45][37].
- **Transport.** WebRTC rooms on a self-hosted SFU or LiveKit Cloud, plus native SIP. A `console` mode offers local single-session testing with audio or text [44].
- **Platform.** Python 3.10–3.14 plus a Node.js port. CI covers Ubuntu, macOS and Windows [46]. Wheels exist for `win_amd64` [52].
- **Strengths.** Opinionated defaults: VAD, turn detector and preemptive generation are on out of the box. Process isolation per job. Mature adapters for realtime and duplex models. Test framework and Agent Console. Telephony.
- **Pain points.**
  - Best-in-class turn and interruption models require LiveKit Cloud or its license. #6033 asks for a self-hostable adaptive interruption model [39][47][51].
  - Room-centric design: multi-participant support (#391) has been open since 2024-06 [51].
  - Echo when using speakers (#315) [51].
  - Many turn-handling knobs, with config bugs such as #7343, where a dict setting was ignored [51].

### 3.3 OpenAI: Agents SDK voice paths, Realtime, GPT-Live

- **Agents SDK** (MIT; Python 29.7k★ at v0.22.3, JS 3.9k★ at v0.18.0) [13] has two voice paths:
  1. **Realtime:** `RealtimeAgent` → `RealtimeRunner` → `RealtimeSession`. Events include `audio_interrupted`, `tool_approval_required` and `guardrail_tripped`. Transports are WebSocket (default) and SIP. `RealtimePlaybackTracker` truncates at the actual playback position. The model is chosen per session, not per agent [11].
  2. **Chained:** `VoicePipeline(workflow=SingleAgentVoiceWorkflow(agent))` running STT → agent → TTS. The SDK "does not provide any built-in interruption handling for `StreamedAudioInput`" [12].
- **GPT-Live** has its own endpoint (`v1/live/sessions`). It accepts audio and text, not image or video. Pricing is $0.05/min billed per second; Tier-1 accounts get 25 concurrent sessions [6]. There are two delegation modes, and your application still runs your functions and enforces permissions [5]. I found no `gpt-live-1` reference in the Agents SDK repos via code search *(unverified; the search index may lag)*.

### 3.4 Hugging Face speech-to-speech (v1.0, 2026-09-06)

- **Architecture.** A cascade of four stages (Silero VAD v5 → STT → LLM → TTS), each "running in its own thread and connected by queues". Backends are chosen with CLI flags. It serves the OpenAI Realtime core events at `/v1/realtime` over WebSocket and WebRTC, and has an optional LLM proxy for side tasks [60].
- **Local models.** Defaults are Parakeet TDT for STT and Qwen3-TTS.
  - **Apple Silicon, fully local:** MLX, about 7.5 GB of weights.
  - **NVIDIA, fully local:** plan for 24 GB of VRAM.
  - **Other backends:** STT and TTS (Whisper variants, Qwen3-ASR, Paraformer, Kokoro, Pocket TTS, OmniVoice, ...) and any OpenAI-compatible LLM (vLLM, llama.cpp) [60].
- **Deployment.** The README says it is the conversation backend for "thousands of Reachy Mini robots" [60].
- **Gaps.** CI runs on Ubuntu and macOS only. There's no echo cancellation (headphones are advised), no telephony, and the design is server/CLI-first rather than library-first [60].

### 3.5 Kyutai Unmute, Moshi and DSM

- **Unmute (MIT).**
  - **Architecture:** the browser talks to a Python backend over WebSocket, using a protocol based on the OpenAI Realtime API. The backend streams audio to Kyutai's Rust STT and TTS servers and calls any text LLM (vLLM or OpenRouter). The STT's semantic VAD decides when to respond [62][63].
  - **Performance:** TTS latency is about 750 ms on a single L40S and about 450 ms on unmute.sh, which splits services across 3 GPUs. The STT server handles 64 streams at 3× real-time on an L40S [62][63].
  - **Requirements:** an x86_64 CUDA GPU with 16 GB+, on Linux or WSL. There's no native Windows or macOS support [62].
- **Moshi** (11.1k★) is the open full-duplex foundation [64]. PersonaPlex is built on it [66].

### 3.6 TEN Framework (Agora)

- **Architecture.** A graph of "extensions". Agora's docs name C++, Python and Node.js as extension languages; Go and Rust also appear in the repository [57][54].
  - **Message types:** four (`cmd`, `data`, `audio_frame`, `video_frame`), routed by name matching.
  - **Threading:** "extension groups" map to threads, with ownership transfer and copy semantics [57].
  - **Tooling:** the TMAN Designer visual graph editor. Examples cover cascaded and realtime assistants, SIP, ESP32 and avatars [54].
- **Requirements.** The default examples need an Agora App ID and certificate, plus Docker [54]. Python 3.10 only; Linux (x64/arm64), macOS (x64/arm64) and Windows x64 [56]. 11.1k★, version 0.11.73.
- **License.** Apache-2.0 plus Agora conditions, notably no hosting on end-user devices [55]. That rules it out as a dependency for local-first apps.

### 3.7 FastRTC (Gradio)

- **The idea:** "Turn any python function into a real-time audio and video stream over WebRTC or WebSockets": `Stream(handler=ReplyOnPause(fn), modality="audio", mode="send-receive")`. Your generator receives the whole utterance and yields audio chunks.
- **Built-ins:** `can_interrupt`; `StreamHandler.receive/emit` for duplex; `.ui.launch()` for a Gradio UI; `.mount(app)` for FastAPI; and `fastphone()` for a temporary phone number [61].
- **Maturity:** MIT, 4.6k★, about 10.9k downloads/month, last release 0.0.34 [53][61].
- **Lesson:** the best ergonomics for the simple case. Orchestration is thin, though: no context management, tool calling or production telephony.

### 3.8 Vocode and Bolna

- **Vocode** (MIT, 3.8k★): `StreamingConversation(input, output, transcriber, agent, synthesizer)` with pydantic configs per provider, plus telephony and Zoom dial-in. Last commit 2024-11-15, and the README is looking for maintainers [58].
- **Bolna** (MIT, 770★): telephony-first. Agents are JSON "tasks" whose `toolchain` runs pipelines such as `["transcriber","llm","synthesizer"]` in parallel. The local setup is Docker Compose with a telephony server, ngrok and Redis. The hosted API and UI are closed. It releases often and is "actively looking for maintainers" [59].

### 3.9 Ultravox (Fixie)

- **Model:** a multimodal projector maps audio straight into an open-weight LLM (Llama, Mistral, Gemma). It "takes in audio and emits streaming text", which makes it a half-cascade model. The default model is built on Llama 3.3 70B; v0.7 shipped in December 2025. MIT, 4.6k★ [65].
- **Platform:** the hosted "Ultravox Realtime" platform has plugins in both Pipecat and LiveKit [14][35].

### 3.10 New entrants, 2025–2026

| Project | Since | License / ★ | What's notable |
|---|---|---|---|
| GetStream **Vision Agents** | 2025-08 | Apache-2.0 / 8.1k | Video-first. `Agent(edge=getstream.Edge(), llm=gemini.Realtime(fps=10), processors=[YOLO...])`. Smart Turn and Vogent turn detection; Prometheus and K8s. The default edge network needs a Stream API key [69] |
| **Dograh** | 2025-09 | BSD-2 / 5.7k | Self-hostable Vapi/Retell alternative with a visual workflow builder and MCP editing. Runs on a Pipecat fork (git submodule) [70] |
| Cartesia **Line** | 2025-08 | Apache-2.0 / 106 | "Brings voice to your text agents": `VoiceAgentApp(get_agent=...)`, with LiteLLM-backed `LlmAgent`. Cartesia hosts orchestration and deployment [71] |
| AWS Strands **BidiAgent** | 2025–26 | Apache-2.0 (SDK) *(unverified)* | Persistent bidirectional stream with pluggable I/O channels `run(inputs=[...], outputs=[...])`. Supports Nova Sonic, OpenAI Realtime and Gemini Live. Python-only, experimental [72] |
| Google **ADK live** | 2025–26 | Apache-2.0 *(unverified)* | `run_live()` + `LiveRequestQueue` + `RunConfig(StreamingMode.BIDI)` on the Gemini Live API. Adds session resumption, persistence and multi-agent transfer [73] |
| NVIDIA **PersonaPlex** | 2026-01 | MIT code / 10.6k | 7B full-duplex model built on Moshi, with role and voice prompts. The LiveKit plugin is experimental: no tools, no transcripts, no history [66][37] |

### 3.11 Hosted platforms

- **Vapi:** documents its pipeline as "User Audio → VAD → Transcription → Start Speaking Decision → LLM → TTS → waitSeconds → Assistant Audio". Behavior is set through declarative plans [74]:
  - **`startSpeakingPlan`:** smart-endpointing providers (LiveKit, Vapi, Krisp, Deepgram Flux, AssemblyAI) and fallback punctuation timers (0.1 / 1.5 / 0.5 s).
  - **`stopSpeakingPlan`:** `numWords`, `voiceSeconds` and `backoffSeconds`.
- **Retell:** a proprietary turn-taking model at about 600 ms that weighs "prosody, semantic completion, and adaptive pacing" (vendor claim) [75]. Its Custom-LLM WebSocket is a clean "bring your own brain" contract [76].
- **Deepgram Voice Agent API:** one WebSocket that bundles listen/think/speak, with bring-your-own LLM and TTS [78].
- **Bland and Synthflow:** I found no public architecture write-ups on their blog indexes [77].
- **Market context:** a16z describes the stack as layers (infrastructure/models, horizontal platforms like Vapi and Bland, vertical apps) and says the market is moving "from the infrastructure to application layer" [3].
- **Google and Anthropic guidance:** Google's ADK treats a live agent as an ordinary ADK agent, with the same agent, tool and session abstractions, whose connection "today ... runs on the Gemini Live API" [73]. Anthropic's docs index had no voice-agent guide on 2026-09-24. Its general latency advice is to pick a faster model (Claude Haiku 4.5), stream, and trim prompt and output tokens [79].

---

## 4. Side-by-side comparison

### 4.1 Maturity and licensing (2026-09-24)

| Framework | License | ★ | Latest release | 1.0 date | PyPI downloads (last month) | Python | CI OSes / platform notes |
|---|---|---|---|---|---|---|---|
| Pipecat | BSD-2 | 15.8k | v1.11.0 (09-18) | 2026-04-14 | ~1.04M | ≥3.11 | Ubuntu |
| LiveKit Agents | Apache-2.0 | 14.3k (+0.9k JS) | 1.8.2 (09-15) | 2025-04-10 | ~3.25M | 3.10–3.14 | Ubuntu/macOS/Windows |
| HF speech-to-speech | Apache-2.0 | 13.3k | v1.0.0 (09-06) | 2026-09-06 | ~2.7k | ≥3.10 | Ubuntu/macOS |
| TEN Framework | Apache-2.0 + Agora conditions | 11.1k | 0.11.73 (09-22) | — | n/a | 3.10 only | not checked |
| Vision Agents | Apache-2.0 | 8.1k | v0.6.9 (08-13) | — | ~2.7k | ≥3.10 | not checked |
| Dograh | BSD-2 | 5.7k | v1.47.0 (09-15) | — | n/a | — | not checked |
| FastRTC | MIT | 4.6k | 0.0.34 (2025-11-24) | — | ~10.9k | ≥3.10 | — |
| Vocode | MIT | 3.8k | 0.1.113 on PyPI | — | ~0.7k | ≥3.10,<4 | dormant since 2024-11 |
| Unmute | MIT | 1.5k | none | — | — | — | runs on Linux/WSL only |
| Bolna | MIT | 0.8k | 0.10.259 (09-24) | — | ~9.2k | ≥3.10 | — |

### 4.2 Abstractions and capabilities

| | Core abstraction | Turn detection (default) | Barge-in | S2S/duplex | Local models | Local AEC |
|---|---|---|---|---|---|---|
| Pipecat | Frames → processors → pipeline; workers on a bus | Silero + Smart Turn v3 (local) | Frame broadcast + flush | Realtime + GPT-Live | Broad | No [28] |
| LiveKit | `AgentSession` + `Agent` + nodes; job processes | Audio TurnDetector (cloud v1 / local mini) | SpeechHandle, adaptive (cloud), resume | Realtime, half-cascade, GPT-Live, PersonaPlex | Thin | Yes, console [46] |
| OpenAI SDK | `RealtimeSession` / `VoicePipeline` | Server/semantic VAD | Server cancel + truncate | Native | None | Browser |
| HF s2s | Threaded stage queues + Realtime server | Silero VAD | Realtime `interrupt_response` | Via audio-input LLM | Broad (MLX/CUDA) | No [60] |
| Unmute | Backend orchestrating STT/LLM/TTS servers | Kyutai semantic VAD | Yes | No | Kyutai models (GPU) | Browser |
| TEN | Extension graph (multi-language) | TEN VAD / turn detection | Yes | Realtime examples | Some | RTC client |
| FastRTC | Handler function per stream | Silero pause detection | `can_interrupt` | Via handler | User-supplied | Browser |

---

## 5. Recurring pain points (with evidence)

1. **Silent failures.** Providers failing without errors (#2876). Smart Turn breaking at 8 kHz (#3844). Pipelines stalling with no error (#721) [31].
2. **Concurrency and process model.** Pipecat freezes with thread pools (#1912) [31]. LiveKit isolates each job in its own process at the cost of memory, and uses prewarm to hide load time [44].
3. **Features locked to a cloud.** Adaptive interruption and the full v1 turn detector run on LiveKit's hosted inference. Usage is unlimited or free only for agents deployed on LiveKit Cloud, with limited allowances elsewhere, and self-hosted agents fall back to VAD or v1-mini (#6033) [38][39][51]. TEN's default examples assume Agora [54].
4. **Echo and self-interruption on local devices.** LiveKit #315 [51]; HF s2s tells users to wear headphones [60].
5. **Multi-party calls.** LiveKit #391 open since 2024; Pipecat #3218 [31][51].
6. **API churn.** Pipecat 1.0 removed many APIs; Flows moved into core [15][24].
7. **Too many knobs.** Turn handling alone exposes dozens of options across endpointing, interruption and preemption [40]. Vapi exposes several "plans" [74].
8. **Platform and dependency friction.** No Windows wheels for Daily, no macOS or Windows for Unmute, Python 3.10 only for TEN, Docker-first local setups (TEN, Bolna, Unmute) [52][62][56][59].
9. **Switching costs.** Third-party comparisons say the pipeline definition is the main lock-in, and advise keeping the "intelligence layer framework-independent" [82]. They rate Pipecat, LiveKit and TEN as production-ready and flag smaller projects as maintenance risks [80]. They frame the decision as roll-your-own vs open-source framework vs managed platform [81].

---

## 6. Implications for voice-agent-next

### 6.1 Core abstractions to adopt

1. **One typed, timestamped event bus with two priority lanes.**
   - **Lanes:** a control lane for VAD, turn and interruption events and cancellations, and an ordered data lane. This keeps Pipecat's best property [18].
   - **Epochs:** every event carries a monotonic timestamp, a `turn_id` and a `response_id` epoch. Cancellation then means "discard anything older than epoch N", the approach Retell's `response_id` and OpenAI's item IDs take [76][10]. It replaces the pattern of "broadcast then hope every queue flushes".
2. **An `Engine` interface that doesn't care about the architecture.** `Cascade(stt, llm, tts)`, `HalfCascade(audio_llm, tts)`, `Realtime(model)` and `Duplex(model, backend=...)` all implement one session contract. Each engine declares its capabilities: user transcripts, exact scripted speech, text-only output, server-side turns, truncation, tool calling and duplex. LiveKit's comparison table [34] and its GPT-Live caveats [36] show these differences have to be explicit, not buried in code. It's also what makes cross-architecture benchmarking possible.
3. **Turn-taking as its own pluggable stack.**
   - **Interfaces:** `VAD`, `EndOfTurnDetector` and `InterruptionPolicy`, all emitting *probabilistic* events: `StartOfTurn`, `EagerEndOfTurn`, `TurnResumed`, `EndOfTurn`, `Backchannel` (a Flux-like vocabulary [78]).
   - **Built-in, permissively licensed pieces:** Silero, Smart Turn [29], LLM markers [28], STT-provided turn events and provider server VAD.
   - **Excluded:** anything under LiveKit or TEN model licenses [47][55].
   - **Calibration:** per-language thresholds, checked against eot-bench [50].
4. **Speculation as a built-in feature.** Preemptive LLM (and optionally TTS) generation on eager end-of-turn, with a budget, a retry cap and metrics for wasted compute. Use LiveKit's defaults as a starting point [40].
5. **Delegation as a first-class feature ("talker/thinker").** `delegate(task)` runs background work whose lifetime and context rendering are *independent* of interruption settings. Pipecat #4657 asks for exactly this separation [31]. Results come back as context events, and the talker can speak progress updates. Any engine should be able to use it, and any external agent framework can serve as the backend, generalizing GPT-Live's client delegation [5].
6. **One authoritative context holding only what the user heard.** Commit assistant text by word or playback timestamps [22][1]. Allow edits only through transactions or transform events, never shared mutable references. That's the race Pipecat 1.0 had to fix [15].
7. **Transports as adapters.**
   - **Local audio:** with echo cancellation, noise suppression and AGC through WebRTC APM, following LiveKit's console design [46].
   - **Network:** WebSocket, WebRTC (aiortc-class P2P) and telephony serializers.
   - **An OpenAI-Realtime-compatible server,** so existing clients and the OpenAI Agents SDK can talk to *any* engine, as HF s2s does [60].
   - **Optional:** an RTVI adapter to reuse Pipecat clients [26].
8. **Process model.**
   - In-process for development and local use.
   - An optional session-per-process worker pool with prewarm for servers [44].
   - Heavy local models run as a shared inference process, or behind an OpenAI-compatible server such as Kokoro-FastAPI or speaches [85], so they aren't loaded once per session. LiveKit already reaches local Kokoro this way [45].

### 6.2 Things to avoid

- Making ordering and forwarding the user's job. Correct wiring should be the default, and the graph should be checked when the session starts.
- Features that depend on a cloud or on a specific framework's license.
- Hard-coding one vendor's transport or SFU into the core.
- Configuration knobs without presets. Offer named profiles like `snappy`, `patient` and `telephony`, compiled down to explicit settings.

### 6.3 API ergonomics (proposal sketch, not an existing API)

```python
from voice_agent_next import Agent, Session, engines, turns, transports

agent = Agent(instructions="You are a concise assistant.", tools=[get_weather])

session = Session(
    agent,
    engine=engines.Cascade(
        stt="local/parakeet-tdt",
        llm="openai-compat/qwen3-4b@http://localhost:8080/v1",
        tts="local/kokoro",
    ),  # or engines.Realtime("openai/gpt-realtime-2.1")
    # or engines.Duplex("openai/gpt-live-1", backend=agent)
    turns=turns.preset("balanced"),  # Silero + Smart Turn + backchannel filter + preemption
    transport=transports.LocalAudio(aec=True),  # WebRTC(), WebSocket(), Twilio(), RealtimeServer()
)
await session.run()
```

What makes this pleasant:

- **Simple things first:** FastRTC-level simplicity for the "just talk to it" case [61].
- **Opinionated defaults:** like LiveKit's AgentSession [38].
- **Readable model specs:** `provider/model` strings [33].
- **Escape hatches:** override hooks and nodes (LiveKit-style [41]) before dropping down to raw processors.
- **Text-only mode:** for tests and evals [25].
- **Loud, typed errors:** capability and sample-rate validation when the session starts.

### 6.4 Gaps we can fill (differentiators)

1. **The same local experience on Linux, macOS and Windows,** with a CI matrix, bundled echo cancellation and hardware-aware backends: MLX on Apple, CUDA/ONNX/CPU elsewhere. Of the frameworks whose CI I checked, only LiveKit tests on Windows, and it has thin local model support. No framework combines the two (§4).
2. **A benchmark harness that works across architectures.** Replay the same recorded or synthetic callers through a cascade, a half-cascade, a realtime model and a duplex model, and report:
   - voice-to-voice p50/p95/p99 measured at playback;
   - false-cutoff rate vs endpointing delay;
   - barge-in latency;
   - context accuracy after interruptions;
   - tool-call correctness;
   - cost per minute.

   Build on eot-bench and stt-benchmark, Full-Duplex-Bench and aiewf-eval [50][30][68][86], and follow OpenAI's Crawl/Walk/Run staging [4].
3. **A fully open turn-taking and interruption stack.** It fills the self-hosting gap users are asking for [51].
4. **Delegation that works with any model**, not just GPT-Live.
5. **Standards-based serving:** OpenAI Realtime compatibility plus RTVI.

### 6.5 Open questions

- How should duplex models share an event model with turn-based engines, given that items arrive after the audio [35]?
- Should Smart Turn be fine-tuned per language or per domain (for example, number dictation)?
- Which local Windows audio stack should we use (PortAudio or WASAPI) to get reliable echo cancellation?

---

## 7. Sources

1. *Voice AI & Voice Agents: An Illustrated Primer* (Daily/Pipecat team). Originally February 2025, updated June 2026. https://voiceaiandvoiceagents.com/
2. Latent Space, "OpenAI Realtime API: The Missing Manual", 2024-11-21. https://www.latent.space/p/realtime-api
3. a16z, "AI Voice Agents: 2025 Update" (O. Moore), 2025-01-29. https://a16z.com/ai-voice-agents-2025-update/
4. OpenAI, "Voice agents" guide (accessed 2026-09-24). https://developers.openai.com/api/docs/guides/voice-agents
5. OpenAI, "Getting started with GPT-Live". https://developers.openai.com/api/docs/guides/live
6. OpenAI, "GPT-Live 1" model page. https://developers.openai.com/api/docs/models/gpt-live-1
7. OpenAI, "Build more natural voice experiences with GPT-Live-1 in the API", 2026-09-10. Returned HTTP 403 to our fetcher; figures taken via [8]. https://openai.com/index/introducing-gpt-live-1-in-the-api/
8. Unite.AI, "OpenAI's GPT-Live-1 Arrives in the API at $0.05 Per Minute", 2026-09-10. https://www.unite.ai/openais-gpt-live-1-arrives-in-the-api-at-0-05-per-minute/
9. OpenAI, "Voice activity detection (VAD)". https://developers.openai.com/api/docs/guides/realtime-vad
10. OpenAI, "Realtime conversations" (interruption and truncation). https://developers.openai.com/api/docs/guides/realtime-conversations
11. OpenAI Agents SDK (Python), Realtime guide. https://openai.github.io/openai-agents-python/realtime/guide/
12. OpenAI Agents SDK (Python), Voice pipeline. https://openai.github.io/openai-agents-python/voice/pipeline/
13. GitHub: openai/openai-agents-python and openai/openai-agents-js. https://github.com/openai/openai-agents-python · https://github.com/openai/openai-agents-js
14. GitHub: pipecat-ai/pipecat (README, metadata). https://github.com/pipecat-ai/pipecat
15. Pipecat v1.0.0 release notes, 2026-04-14. https://github.com/pipecat-ai/pipecat/releases/tag/v1.0.0
16. Pipecat v1.9.0 and v1.11.0 release notes, 2026-09-11 / 2026-09-18. https://github.com/pipecat-ai/pipecat/releases/tag/v1.9.0 · https://github.com/pipecat-ai/pipecat/releases/tag/v1.11.0
17. Pipecat docs, "Overview of Pipecat". https://docs.pipecat.ai/pipecat/learn/overview
18. Pipecat docs, "Pipeline & Frame Processing". https://docs.pipecat.ai/pipecat/learn/pipeline
19. Pipecat docs, "Your First Agent". https://docs.pipecat.ai/pipecat/learn/your-first-agent
20. Pipecat docs, "The Worker Bus". https://docs.pipecat.ai/pipecat/fundamentals/agent-bus
21. Pipecat docs, "Speech Input & Turn Detection". https://docs.pipecat.ai/pipecat/learn/speech-input
22. Pipecat docs, "Interruptions". https://docs.pipecat.ai/pipecat/fundamentals/interruptions
23. Pipecat docs, "Transports". https://docs.pipecat.ai/pipecat/learn/transports
24. Pipecat docs, "Migrating from pipecat-ai-flows". https://docs.pipecat.ai/pipecat/migration/flows
25. Pipecat docs, "Pipecat Evals". https://docs.pipecat.ai/pipecat/evals/overview
26. Pipecat docs, "The RTVI Standard" (v1.0, June 2025). https://docs.pipecat.ai/client/rtvi-standard
27. Pipecat docs index (service catalog). https://docs.pipecat.ai/llms.txt
28. Pipecat source and CI: turn-completion mixin, local audio transport, tests workflow. https://github.com/pipecat-ai/pipecat/blob/main/src/pipecat/turns/user_turn_completion_mixin.py · https://github.com/pipecat-ai/pipecat/blob/main/src/pipecat/transports/local/audio.py · https://github.com/pipecat-ai/pipecat/blob/main/.github/workflows/tests.yaml
29. Smart Turn README and model. https://github.com/pipecat-ai/smart-turn · https://huggingface.co/pipecat-ai/smart-turn-v3
30. pipecat-ai/stt-benchmark. https://github.com/pipecat-ai/stt-benchmark
31. Pipecat issues #3116, #1912, #3844, #2876, #3218, #3757, #4657, #721. https://github.com/pipecat-ai/pipecat/issues/3116 · https://github.com/pipecat-ai/pipecat/issues/1912 · https://github.com/pipecat-ai/pipecat/issues/3844 · https://github.com/pipecat-ai/pipecat/issues/2876 · https://github.com/pipecat-ai/pipecat/issues/3218 · https://github.com/pipecat-ai/pipecat/issues/3757 · https://github.com/pipecat-ai/pipecat/issues/4657 · https://github.com/pipecat-ai/pipecat/issues/721
32. Archived repos pipecat-flows and pipecat-subagents. https://github.com/pipecat-ai/pipecat-flows · https://github.com/pipecat-ai/pipecat-subagents
33. GitHub: livekit/agents (README, metadata, release notes incl. livekit-agents@1.8.1 of 2026-09-10) and livekit/agents-js. https://github.com/livekit/agents · https://github.com/livekit/agents/releases/tag/livekit-agents%401.8.1 · https://github.com/livekit/agents-js
34. LiveKit docs, "Pipeline types". https://docs.livekit.io/agents/models/pipelines/
35. LiveKit docs, "Realtime models overview". https://docs.livekit.io/agents/models/realtime/
36. LiveKit docs, "OpenAI GPT-Live plugin guide". https://docs.livekit.io/agents/models/realtime/plugins/gpt-live/
37. LiveKit docs, "NVIDIA PersonaPlex plugin guide". https://docs.livekit.io/agents/models/realtime/plugins/personaplex/
38. LiveKit docs, "Turn detector". https://docs.livekit.io/agents/logic/turns/turn-detector/
39. LiveKit docs, "Adaptive interruption handling". https://docs.livekit.io/agents/logic/turns/adaptive-interruption-handling/
40. LiveKit docs, "Turn-taking tuning" and "Turns overview". https://docs.livekit.io/agents/logic/turns/tuning/ · https://docs.livekit.io/agents/logic/turns/
41. LiveKit docs, "Pipeline nodes and hooks". https://docs.livekit.io/agents/logic/nodes/
42. LiveKit docs, "Agent sessions" and "Speech & audio". https://docs.livekit.io/agents/logic/sessions/ · https://docs.livekit.io/agents/multimodality/audio/
43. LiveKit docs, "Workflows". https://docs.livekit.io/agents/logic/workflows/
44. LiveKit docs, "Agent server options" and "Startup modes". https://docs.livekit.io/agents/server/options/ · https://docs.livekit.io/agents/server/startup-modes/
45. LiveKit docs, "Kokoro TTS plugin guide" and agents docs index. https://docs.livekit.io/agents/models/tts/kokoro/ · https://docs.livekit.io/agents/llms.txt
46. LiveKit source: console-mode APM and CI matrix. https://github.com/livekit/agents/blob/main/livekit-agents/livekit/agents/cli/_legacy.py · https://github.com/livekit/agents/blob/main/.github/workflows/tests.yml
47. Hugging Face: livekit/turn-detector model card and LICENSE. https://huggingface.co/livekit/turn-detector
48. LiveKit blog, "Solving end-of-turn detection", 2026-06-17. https://livekit.com/blog/solving-end-of-turn-detection
49. LiveKit blog, "Sequential Pipeline Architecture for Voice Agents" (J. Hall), 2026-03-23. https://livekit.com/blog/sequential-pipeline-architecture-voice-agents
50. livekit/eot-bench. https://github.com/livekit/eot-bench
51. LiveKit issues #6033, #391, #315, #1924, #7343. https://github.com/livekit/agents/issues/6033 · https://github.com/livekit/agents/issues/391 · https://github.com/livekit/agents/issues/315 · https://github.com/livekit/agents/issues/1924 · https://github.com/livekit/agents/issues/7343
52. PyPI file listings and release histories. https://pypi.org/project/daily-python/#files · https://pypi.org/project/livekit/#files · https://pypi.org/project/livekit-agents/#history · https://pypi.org/project/pipecat-ai/#history
53. pypistats, last-month downloads (2026-09-24). https://pypistats.org/packages/pipecat-ai · https://pypistats.org/packages/livekit-agents · https://pypistats.org/packages/fastrtc · https://pypistats.org/packages/bolna · https://pypistats.org/packages/vocode · https://pypistats.org/packages/speech-to-speech · https://pypistats.org/packages/vision-agents
54. TEN Framework README. https://github.com/TEN-framework/ten-framework
55. TEN licenses (framework, VAD, turn detection). https://github.com/TEN-framework/ten-framework/blob/main/LICENSE · https://github.com/TEN-framework/ten-vad · https://github.com/TEN-framework/ten-turn-detection
56. TEN quick start (platforms, Python 3.10). https://github.com/TEN-framework/ten-framework/blob/main/docs/getting-started/quick-start.md
57. Agora docs, TEN core concepts and message system. https://docs.agora.io/en/ai/ten-agent/core-concepts · https://docs.agora.io/en/ten-framework/architecture/message-system
58. GitHub: vocodedev/vocode-core. https://github.com/vocodedev/vocode-core
59. GitHub: bolna-ai/bolna (README, `bolna/assistant.py`). https://github.com/bolna-ai/bolna
60. GitHub: huggingface/speech-to-speech (README, v1.0.0, CI). https://github.com/huggingface/speech-to-speech
61. GitHub: gradio-app/fastrtc (README, audio guide). https://github.com/gradio-app/fastrtc · https://github.com/gradio-app/fastrtc/blob/main/docs/userguide/audio.md
62. GitHub: kyutai-labs/unmute. https://github.com/kyutai-labs/unmute
63. GitHub: kyutai-labs/delayed-streams-modeling. https://github.com/kyutai-labs/delayed-streams-modeling
64. Défossez et al., "Moshi: a speech-text foundation model for real-time dialogue", arXiv:2410.00037, 2024. https://arxiv.org/abs/2410.00037
65. GitHub: fixie-ai/ultravox. https://github.com/fixie-ai/ultravox
66. NVIDIA PersonaPlex repo and paper (arXiv:2602.06053, February 2026). https://github.com/NVIDIA/personaplex · https://arxiv.org/abs/2602.06053
67. "KAME: Tandem Architecture for Enhancing Knowledge in Real-Time Speech-to-Speech Conversational AI", arXiv:2510.02327, 2025. https://arxiv.org/abs/2510.02327
68. Full-Duplex-Bench (arXiv:2503.04721, March 2025) and ECHO (arXiv:2609.17360, September 2026). https://arxiv.org/abs/2503.04721 · https://arxiv.org/abs/2609.17360
69. GitHub: GetStream/Vision-Agents. https://github.com/GetStream/Vision-Agents
70. GitHub: dograh-hq/dograh (README, `.gitmodules`). https://github.com/dograh-hq/dograh
71. GitHub: cartesia-ai/line. https://github.com/cartesia-ai/line
72. Strands Agents, "Build a realtime voice agent" (BidiAgent). https://strandsagents.com/docs/user-guide/sdk/bidirectional-streaming/
73. Google ADK, "Live and voice agents". https://adk.dev/live/
74. Vapi docs, "Voice pipeline configuration". https://docs.vapi.ai/customization/voice-pipeline-configuration
75. Retell AI, "Turn-Taking in Voice AI: The Hidden Problem That Breaks Most Demos", 2026-08-11. https://www.retellai.com/blog/turn-taking-voice-ai-hidden-problem
76. Retell docs, Custom LLM integration overview. https://docs.retellai.com/integrate-llm/overview
77. Bland blog index; Synthflow blog index (checked 2026-09-24). https://www.bland.ai/blog · https://synthflow.ai/blog
78. Deepgram docs: Flux quickstart and Voice Agent API. https://developers.deepgram.com/docs/flux/quickstart · https://developers.deepgram.com/docs/voice-agent
79. Anthropic docs, "Reducing latency" and docs index. The index had no voice-agent-specific guide as of 2026-09-24; the latency guidance recommends faster models (Claude Haiku 4.5), streaming and fewer tokens. https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/reduce-latency · https://docs.claude.com/llms.txt
80. Techsy, "6 Best Open-Source Voice Agent Frameworks (2026)", 2026-07-15. https://techsy.io/en/blog/best-open-source-voice-agent-frameworks
81. Soniox Voice AI Wiki, "Voice agent frameworks", updated 2026-07-02. https://soniox.com/wiki/voice-agent-frameworks
82. Chanl, "Pipecat vs LiveKit: the trade-offs that lock you in", 2026-04-03. https://www.channel.tel/blog/pipecat-vs-livekit-voice-framework-decision
83. L. H. Thuan, "Pipecat Voice Agent in Production", 2026-03-16 (secondary; issue numbers re-verified on GitHub). https://luonghongthuan.com/en/blog/pipecat-voice-agent-production-scalable-guide/
84. GitHub: KoljaB/RealtimeVoiceChat. https://github.com/KoljaB/RealtimeVoiceChat
85. OpenAI-compatible local speech servers: Kokoro-FastAPI, speaches. https://github.com/remsky/Kokoro-FastAPI · https://github.com/speaches-ai/speaches
86. kwindla/aiewf-eval (30-turn voice-agent LLM eval referenced by [1]). https://github.com/kwindla/aiewf-eval
