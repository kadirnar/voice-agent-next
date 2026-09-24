# 02 — Native Speech-to-Speech Models and Realtime APIs

*Research snapshot: 2026-09-24. Scope: cloud realtime voice APIs and open-weights speech-to-speech (S2S) / audio LLMs, with the aim of building one engine abstraction for `voice-agent-next`. Every fact is cited as [n]. Items marked **(unverified)** come from secondary sources or could not be checked against a primary source. Per-minute costs marked **(derived)** are my own arithmetic from the published token prices.*

---

## 1. Executive summary

- **The cloud frontier moved a lot in 2026.** OpenAI shipped **GPT-Realtime-2** (May 2026), which has configurable reasoning (`minimal`→`xhigh`) and a 128k context [3][21]. It followed with **GPT-Realtime-2.1 / 2.1-mini** (2026-07-06) [1][4][5] and **GPT-Live-1** (GA 2026-09-10). GPT-Live-1 is a full-duplex "voice layer" priced at **$0.05/min**. It hands reasoning and tools off to a backend agent, and it uses a **new protocol** (`/v1/live/sessions`) that is *not* the Realtime API [1][16][18][19]. Google shipped **Gemini 3.1 Flash Live** (2026-03-26) and then **Gemini 3.8 Live / 3.8 Live Extended Thinking** (2026-09-15). Google says 3.8 Live ranks #1 on Artificial Analysis' S2S leaderboard [32][33][30].
- **The OpenAI Realtime protocol is the de facto lingua franca.** Azure OpenAI and xAI Grok Voice Agent implement it natively [24][39]. Azure Voice Live, Alibaba Qwen-Omni-Realtime, vLLM `/v1/realtime`, Kyutai Unmute, Speaches, LocalAI and LiteLLM speak it or a close dialect [22][42][71][57][98][99][100]. **One well-built OpenAI-Realtime client with "compat profiles" covers about 10 backends.**
- **Gemini Live (BidiGenerateContent), Amazon Nova 2 Sonic (Bedrock bidirectional stream), GPT-Live, Moshi/PersonaPlex (binary Opus protocol) and Hume / Ultravox / ElevenLabs / Deepgram each use their own protocol** [28][38][18][55][45][48][50][51]. They need dedicated adapters.
- **Audio formats differ:**
  - OpenAI: PCM16 at **24 kHz** in and out [8]
  - Gemini: **16 kHz in / 24 kHz out** [25]
  - Nova 2 Sonic: 16 kHz in / 24 kHz out [36]
  - Qwen-Omni-Realtime: 16 kHz in / 24 kHz out [42]
  - Moshi: 24 kHz Opus in Ogg [55]
  - xAI: flexible, 8–48 kHz [39]

  An internal canonical format plus a high-quality resampler is mandatory.
- **Session limits vary:**
  - OpenAI Realtime: 60 min [8]
  - Gemini Live: 15 min audio-only without compression, connections of about 10 min, with GoAway and resumption handles [27]
  - Nova 2 Sonic: an **8-minute connection limit** [36]
  - Qwen realtime: 120 min [42]
  - Hume EVI: 30 min [45]

  Reconnection and context carry-over must be first-class features.
- **Pricing converges on roughly $0.02–0.10 per conversation-minute:**
  - GPT-Realtime-2.x audio: $32 / $64 per 1M tokens in / out, about $0.019/min in and $0.077/min out (derived) [3][10]
  - GPT-Realtime-2.1-mini: $10 / $20 per 1M [5]
  - Gemini 3.8 Live: $0.005/min in and $0.018/min out [29]
  - GPT-Live-1: $0.05/min [16]
  - Grok Voice: $0.08/min [41]
  - Ultravox: $0.05/min [49]
- **Open weights now include genuinely full-duplex models:**
  - **NVIDIA PersonaPlex-7B** (2026-01): Moshi-based, commercial-use license [61]
  - Kyutai's RL / RAG Moshi variants (2026) [60][59]
  - **MiniCPM-o 4.5** (2026-02, 9B, Apache-2.0, full-duplex, 11 GB in int4) [72][73]
  - **Realtime-Venus** (2026-09, 9B, Apache-2.0): full-duplex with asynchronous `<delegate>` tool calls [74]
- **The strongest open S2S model, Qwen3-Omni-30B-A3B (Apache-2.0), does not fit on consumer GPUs:** it needs at least 68–79 GB in BF16 [65]. **Qwen3.5-Omni and Qwen3.8-Omni are API-only.** No official weights were found on the Qwen Hugging Face org as of today [44].
- **Best local candidates for a 16 GB GPU:**
  - MiniCPM-o 4.5 in int4 (full-duplex via llama.cpp on 12 GB+) [73]
  - Moshi / PersonaPlex in int8, or with CPU offload [54][62]
  - LFM2.5-Audio-1.5B (tiny, llama.cpp/CPU, English only) [75]
  - A Kyutai STT → LLM → TTS cascade (Unmute needs 16 GB or more) [57]
- **Best local candidates for Apple Silicon:**
  - Moshi via official MLX q4/q8 [54]
  - MiniCPM-o 4.5 via llama.cpp: M3/M4/M5 with 16 GB for half-duplex, M4 Max with 24 GB for full-duplex [73]
  - Community MLX ports of PersonaPlex [64]
- **Architectural trend: "voice front-end + delegated backend".** GPT-Live-1's Responses/client delegation [19], Realtime-Venus `<delegate>` [74], MoshiRAG's async retrieval [59] and NVIDIA-style delegation tokens [97] all follow this pattern. `voice-agent-next` should make **delegation / async tool calls** a first-class concept, not just synchronous function calls.

---

## 2. Taxonomy used in this document

| Class | Definition | Examples |
|---|---|---|
| **Native S2S, full-duplex** | Listens and speaks at the same time; turn-taking is learned by the model | Moshi, PersonaPlex, MiniCPM-o 4.5, Realtime-Venus, GPT-Live-1 |
| **Native S2S, turn-based ("half-duplex")** | Audio in and audio out in one model, but turns are delimited by VAD or the server | gpt-realtime-*, Gemini Live, Nova 2 Sonic, Qwen3-Omni, LFM2.5-Audio, Step-Audio 2 |
| **Speech-in / text-out** | Audio encoder plus LLM; needs a TTS for output | Ultravox, Voxtral, Gemma 4 E2B/E4B/12B, Nemotron 3 Nano Omni |
| **Managed cascades** | STT → LLM → TTS hosted behind one WebSocket | Azure Voice Live (text-model modes), Deepgram Voice Agent, ElevenLabs Agents, Speechmatics, Unmute, Speaches, LocalAI |

---

## 3. Cloud realtime APIs

### 3.1 OpenAI Realtime API (GA)

**Models (all served only on `/v1/realtime`):**

| Model | Released | Context / max out | Text $/1M (in / cached / out) | Audio $/1M (in / cached / out) |
|---|---|---|---|---|
| `gpt-realtime` (`-2025-08-28`) | 2025-08-28 | 32k / 4,096 | 4 / 0.40 / 16 | 32 / 0.40 / 64 [2] |
| `gpt-realtime-mini` (`-2025-10-06`, `-2025-12-15`) | 2025-10 / 12 | 32k / 4,096 | 0.6 / 0.06 / 2.4 | not captured [6] |
| `gpt-realtime-1.5` | n/a | 32k / 4,096 | 4 / 0.40 / 16 | 32 / 0.40 / 64 [7] |
| `gpt-realtime-2` | 2026-05-07 [1][21] | 128k / 32k | 4 / 0.40 / 24 | 32 / 0.40 / 64 [3] |
| `gpt-realtime-2.1` | 2026-07-06 | 128k / 32k | 4 / 0.40 / 24 | 32 / 0.40 / 64 [4][1] |
| `gpt-realtime-2.1-mini` | 2026-07-06 | 128k / 32k | 0.6 / 0.06 / 2.4 | 10 / 0.30 / 20 [5] |

GPT-Realtime-2 notes [21]:
- Reasoning effort: `minimal`, `low` (default), `medium`, `high`, `xhigh`.
- Time to first audio: 1.12 s at minimal, 2.33 s at high.
- Big Bench Audio: 96.6% (high).

GPT-Realtime-2.1 adds better alphanumeric recognition, silence and noise handling, and interruption behaviour [1]. According to the changelog (seen via a search snippet), the Realtime API beta was removed on 2026-05-12 [1]. Clients must use GA event names; third-party dialects still use beta names (see §5).

**Transports**
- **WebSocket:** `wss://api.openai.com/v1/realtime?model=…`. Recommended for server-to-server [8].
- **WebRTC:** `POST /v1/realtime/calls` (SDP offer/answer). Ephemeral keys come from `POST /v1/realtime/client_secrets`. Events travel on the `oai-events` data channel, and the audio format is negotiated by SDP [11].
- **SIP:** `sip:$PROJECT_ID@sip.api.openai.com;transport=tls`, a `realtime.call.incoming` webhook, `POST /v1/realtime/calls/{id}/accept`, and a sideband WebSocket `?call_id=`. DTMF is reported via `transport.dtmf.received` [12][1].

**Event protocol (GA)** [8]
- Client events:
  - `session.update`
  - `input_audio_buffer.append` / `.commit` / `.clear` (append chunks up to 15 MB)
  - `conversation.item.create`
  - `conversation.item.truncate`
  - `response.create` / `response.cancel`
  - `output_audio_buffer.clear`
- Server events:
  - `session.created` / `.updated`
  - `conversation.item.added` / `.done`
  - `input_audio_buffer.speech_started` / `speech_stopped` / `committed`
  - `response.created`, `response.output_item.added`, `response.content_part.added`
  - `response.output_text.delta`
  - `response.output_audio.delta` / `.done`
  - `response.output_audio_transcript.delta` / `.done`
  - `response.function_call_arguments.delta`
  - `response.output_item.done`, `response.done`, `response.cancelled`
  - `rate_limits.updated`, `error`
- Input transcription arrives as `conversation.item.input_audio_transcription.delta` / `.completed`. Ordering across turns is not guaranteed, so match on `item_id` [13].

**Session config (GA shape)** [8]
- `type: "realtime"`, `model`, `output_modalities`, `instructions`, `tools`, `tool_choice`
- `audio.input.format` (`audio/pcm` at `rate: 24000`)
- `audio.output.format` (`audio/pcm` or G.711 `audio/pcmu`)
- `audio.output.voice`: alloy, ash, ballad, coral, echo, sage, shimmer, verse, marin, cedar
- The voice cannot change after the first audio output.
- Azure documents `reasoning.effort` and output-item `phase` values (`commentary` preambles vs `final_answer`) for the 2.x models [24].

**Turn detection** [8][9]
- `server_vad`: `threshold`, `prefix_padding_ms`, `silence_duration_ms`, `create_response`, `interrupt_response`.
  - Defaults 0.5 / 300 / 500 ms per the API reference [14] (seen via search snippet).
  - `idle_timeout_ms` applies to `server_vad` only [14].
- `semantic_vad` with `eagerness` = `low | medium | high | auto` (`auto` = `medium`). This is the default in current guides [8][9].
- `turn_detection: null` gives push-to-talk: the client commits the buffer and sends `response.create`.

**Interruption and truncation** [8]
- Over WebRTC and SIP, the server truncates unplayed audio automatically.
- Over WebSocket, the client must:
  1. Stop playback on `speech_started`.
  2. Track how much audio was actually played.
  3. Send `conversation.item.truncate {item_id, content_index, audio_end_ms}`.

  Transcript text past the cut is also removed.

**Tools** [8]
- Function definitions go in `session.tools` or `response.tools`.
- Calls arrive as `function_call` items (argument deltas, then `response.done`).
- Return results with `conversation.item.create {type: function_call_output, call_id, output}` followed by `response.create`.
- Out-of-band responses (`response.conversation: "none"`) allow side tasks such as classification without polluting history.

**Limits and cost mechanics**
- Maximum session length is **60 min** [8].
- Audio tokenizes at **1 token / 100 ms of input and 1 token / 50 ms of output** [10]. So GPT-Realtime-2.x costs about **$0.0192 per input-minute and $0.0768 per output-minute (derived)**, which matches third-party figures of $1.15 / $4.61 per hour (unverified) [21].
- When context overflows, the oldest items are dropped. `truncation.retention_ratio` below 1 reduces cache-busting [10].
- Every `response.create` re-reads the conversation, so long sessions cost more per minute. Prompt caching ($0.40/1M cached) mitigates this [10].

### 3.2 OpenAI GPT-Live-1 (new "Live" protocol)

GPT-Live-1 went GA on **2026-09-10** [1]. It is described as a full-duplex model that "listens and speaks at the same time" and delegates reasoning and tools to a backend [16][17].

- **Endpoint:** `v1/live/sessions` only [16]. Transports are WebRTC, WebSocket and SIP [17][20].
- **Price:** $0.05 per minute, billed per second. Backend model and tool usage are billed separately [16].
- **Rate limits:** counted in concurrent sessions, from 25 (Tier 1) to 500 (Tier 5) [16].
- **Context:** 128k tokens by default. At the duration limit the session emits `session.closed` with reason `"expired"` [18].
- **Session events** [18]:
  - `session.started`, `session.updated`, `session.closed`
  - `session.usage.updated` (the latest `usage.seconds` is authoritative)
  - `session.input_audio.append` and `session.output_audio.delta` (WebSocket)
  - `session.input_transcript.delta` / `session.output_transcript.delta`, with `start_ms` / `end_ms`
  - `session.input_audio.mute` / `unmute`
  - `session.instructions.append`, `session.thinking.append` (facts, not spoken) and `session.commentary.append` (content the model may paraphrase aloud)
- **Delegation** [19]:
  - **Responses delegation:** `delegation.responses.{model, instructions, tools, tool_choice, parallel_tool_calls, service_tier}`. Backend events arrive wrapped as `response.event`. Tool results go back as `response.item.create` (`function_call_output`) plus `response.create`.
  - **Client delegation:** your agent receives `session.delegation.created` and answers via commentary or thinking appends.
  - **Interrupting speech does not cancel backend work.**
- Community-reported figures: 12 voices, 87% tool-call success, and 0.798 s latency vs 1.41 s for gpt-realtime-2.1 **(unverified)** [20].

### 3.3 Azure OpenAI Realtime and Azure Voice Live

**Azure OpenAI Realtime** [24]
- GA endpoint: `wss://<resource>.openai.azure.com/openai/v1/realtime?model=<deployment>` (no `api-version`). WebSocket, WebRTC and SIP are supported.
- Models: `gpt-realtime-2.1` and `gpt-realtime-2.1-mini` (version `2026-07-07`).
- Audio: PCM16 at 24 kHz.
- `reasoning.effort` is `minimal`–`high`.
- Azure states a **256k** context for 2.x, while OpenAI's pages say 128k. This is a discrepancy to test.
- **2.x does not support the `truncation` session property.**

**Azure Voice Live** [22][23] is a managed, OpenAI-Realtime-compatible superset: its events "mostly match" Azure OpenAI Realtime, and its extensions are optional.
- Extensions: noise suppression, echo cancellation, advanced end-of-utterance detection, Azure semantic VAD, 140+ STT locales, 600+ voices, avatars.
- Models include:
  - native S2S: `gpt-realtime-2.1` (preview), `gpt-realtime-1.5`, `gpt-realtime`, `gpt-realtime-mini`
  - text LLMs wrapped with Azure STT/TTS: `gpt-4.1*`, `gpt-5*`, `gpt-5.6-terra` / `luna`
  - `phi4-mm-realtime` and `azure-realtime`
- Pricing is tiered Pro / Basic / Lite by model. The token-estimation table says OpenAI models use about 10 input and 20 output tokens per second of audio.
- The FAQ lists a 100k TPM default quota, async function calling, MCP support, SDKs for Python, C#, Java and JS, and **no SIP**.

### 3.4 Google Gemini Live API

**Models** [30][26]
- `gemini-3.8-live`: stable, the recommended default.
- `gemini-3.8-live-extended-thinking`: stable; `thinkingLevel` low / medium / high, `includeThoughts`.
- `gemini-3.1-flash-live-preview`: legacy.
- `gemini-2.5-flash-native-audio-preview-12-2025`: restricted to prior users.

**Benchmarks** [33][32]
- 3.1 Flash Live: ComplexFuncBench Audio 90.8%; Audio MultiChallenge 36.1% with thinking.
- 3.8 Live: 97+ languages, asynchronous function calling, and #1 on the Artificial Analysis S2S leaderboard according to Google.

**Protocol** [28][25]
- One stateful WSS connection to `…GenerativeService.BidiGenerateContent`.
- Client messages: `setup` (model, generationConfig, systemInstruction, tools, realtimeInputConfig, sessionResumption, contextWindowCompression, input/outputAudioTranscription, proactivity), then `clientContent` (turns + `turnComplete`), `realtimeInput` (`audio`, `video`, `text`, `activityStart`, `activityEnd`, `audioStreamEnd`) and `toolResponse`.
- Server messages:
  - `setupComplete`
  - `serverContent` (`modelTurn`, `interrupted`, `turnComplete`, `generationComplete`, `inputTranscription`, `outputTranscription`, `waitingForInput`)
  - `toolCall`, `toolCallCancellation {ids}`
  - `goAway {timeLeft}`, `sessionResumptionUpdate`, `usageMetadata`
- Ephemeral tokens are available for browser clients [28].
- The native transport is a stateful WSS connection. WebRTC is available through partner integrations: LiveKit, Pipecat, Fishjam, Agora, Voximplant, Firebase [25].

**Audio** [25][26]
- Input: raw 16-bit PCM at 16 kHz, little-endian (`audio/pcm;rate=16000`).
- Output: 24 kHz.
- Native-audio models return only the `AUDIO` modality. Text requires `output_audio_transcription`.

**VAD** [26][28]
- `automaticActivityDetection {disabled, startOfSpeechSensitivity, endOfSpeechSensitivity, prefixPaddingMs, silenceDurationMs}`.
- Manual mode uses `activityStart` / `activityEnd`.
- `activityHandling` is `START_OF_ACTIVITY_INTERRUPTS` (default) or `NO_INTERRUPTION`, plus `turnCoverage`.
- Affective dialog and proactive audio are v1beta features.

**Tools** [26][28]
- Function calling with `behavior: NON_BLOCKING` (the default for 3.8 Live) or `BLOCKING`.
- Result scheduling: `SILENT`, `WHEN_IDLE`, `INTERRUPTED`.
- The server can cancel pending calls with `toolCallCancellation`.

**Limits** [27][26]
- Audio-only sessions: 15 min; audio + video: 2 min (without compression).
- Connections end after about 10 min, with a preceding `goAway`.
- `sessionResumption` handles stay valid for 2 h.
- `contextWindowCompression {triggerTokens, slidingWindow.targetTokens}` extends sessions.
- Context: 128k for native-audio models.

**Pricing (3.8 Live, 3.8 ET and 3.1 Flash Live)** [29]

| | Text | Audio | Image / video |
|---|---|---|---|
| Input | $0.75/1M | $3.00/1M (**$0.005/min**) | $1.00/1M ($0.002/min) |
| Output | $4.50/1M | $12.00/1M (**$0.018/min**) | n/a |

- There is a free tier.
- Live Translate bills at 25 tokens per second of audio.
- On Vertex / Agent Platform the Live API is documented for "2.5 Flash" [34]. GA timing on Vertex for 3.8 is **(unverified)**.

### 3.5 Amazon Nova 2 Sonic (Bedrock)

**Model** [35][37]
- `amazon.nova-2-sonic-v1:0`, launched 2025-12-02.
- Context: 1M tokens; max output: 64K.
- Regions: us-east-1, us-west-2, eu-north-1, ap-northeast-1.

**Protocol** [35][36][38]
- Uses `InvokeModelWithBidirectionalStream`, an HTTP/2 event stream carrying JSON events.
- Python SDK: the **experimental** `aws_sdk_bedrock_runtime` (awslabs/aws-sdk-python), not boto3 streaming.
- Input events, in order: `sessionStart` (inference config + `turnDetectionConfiguration.endpointingSensitivity` = HIGH / MEDIUM / LOW), `promptStart` (output audio config and voice, e.g. `matthew`), `contentStart` (type TEXT / AUDIO / TOOL; roles such as SYSTEM and USER), `textInput`, `audioInput`, `toolResult`, `contentEnd`, `promptEnd`, `sessionEnd`.
- Output events: `contentStart` (with `generationStage` SPECULATIVE / FINAL), `textOutput`, `audioOutput`, `toolUse`, `contentEnd` (type TOOL triggers tool execution), `completionEnd`.
- In the v1 samples, barge-in is signalled in-band as `{ "interrupted" : true }` inside `textOutput` [38].

**Audio** [36]
- Input: LPCM 16 kHz, 16-bit mono, base64.
- Output: LPCM 24 kHz.

**Features** [37]
- Languages: English (US, UK, India, Australia), French, Italian, German, Spanish, Portuguese, Hindi.
- Polyglot voices, async tool handling, and mixed text/audio input.

**Limits and pricing**
- **8-minute connection limit.** AWS publishes a session-continuation sample [36][37].
- Pricing is on the Bedrock pricing page. Third-party trackers cite about $3 / $12 per 1M speech tokens in / out **(unverified)**.

### 3.6 xAI Grok Voice Agent API

- **Endpoint:** `wss://api.x.ai/v1/realtime`. It is **compatible with OpenAI Realtime clients** if you change the base URL. Ephemeral tokens are available for browsers [39].
- **Models:** `grok-voice-latest` and `grok-voice-think-fast-2.0` [39].
- **Audio** [39]:
  - PCM at 8, 16, 22.05, 24, 32, 44.1 or 48 kHz
  - PCMU / PCMA at 8 kHz
  - Opus at 24 kHz
  - Sent as base64 JSON or binary frames
- **Turn detection:** `server_vad` or null [39].
- **Voices:** 20+ built-in plus custom voices [39].
- **Tools:** `function`, `web_search`, `x_search`, `file_search`, remote `mcp` [39].
- **Reasoning:** `reasoning.effort` is `high` or `none` [40].
- **Differences from OpenAI** [40]:
  - `conversation.item.input_audio_transcription.updated` (cumulative) instead of deltas
  - no `conversation.item.done` or `rate_limits.updated`
  - extensions: `force_message`, pronunciation `replace`, and session resumption via `conversation_id` (expires after 30 min of inactivity)
- **Price:** $0.08/min ($4.80/h) plus $0.004 per text input [41].

### 3.7 Alibaba Qwen-Omni-Realtime (Model Studio / DashScope)

- **Models:** Qwen3.8-Omni-Flash-Realtime, Qwen3.5-Omni-Plus/Flash-Realtime and the older `qwen3-omni-flash-realtime` [42].
- **Endpoints:** `wss://{WorkspaceId}.{region}.maas.aliyuncs.com/api-ws/v1/realtime` over WebSocket and WebRTC [42].
- **Events:** OpenAI-style events (`session.update`, `input_audio_buffer.append/commit`, `response.create`), but server events use **beta-era names** (`response.audio.delta`, `response.text.delta`, `response.audio_transcript.done`) [42].
- **Audio:** 16 kHz PCM16 mono in, 24 kHz PCM out [42].
- **VAD:** `server_vad`, `semantic_vad` or manual [42].
- **Limits:** sessions up to 120 min [42].
- **Features:** function calling, voice cloning, web search [42].
- Qwen3.8-Omni-Flash (2026-09-18) is API-only, with 1M context and function calling [44].

### 3.8 Other voice-agent platforms (mostly not native S2S)

- **Hume EVI** [45][46]
  - Architecture: speech-language model plus optional supplemental LLMs.
  - Versions: EVI 3 (English) and EVI 4-mini (11 languages; requires a supplemental LLM).
  - Protocol: proprietary WSS JSON. Input is `linear16` at a declared rate; output is base64 WAV.
  - Limits: 30-min sessions, 16 MB messages.
  - Price: about $0.04–0.07/min **(unverified)**.
- **Ultravox Realtime** [47][48][49][89]
  - Architecture: speech-native input (Whisper encoder into an LLM) with TTS output.
  - Protocol: WebRTC recommended; a `serverWebSocket` medium streams s16le PCM with configurable `inputSampleRate` / `outputSampleRate` and `PlaybackClearBuffer`.
  - Price: $0.05/min, 5 concurrent calls on pay-as-you-go.
  - Open weights: `ultravox-v0_7-glm-4_6` (MIT, text output only; Big Bench Audio 97.0% with reasoning, VoiceBench 90.75).
- **ElevenLabs Agents** [50]
  - `wss://api.elevenlabs.io/v1/convai/conversation`.
  - Client events: `user_audio_chunk`, `client_tool_result`, `contextual_update`.
  - Server events: `audio`, `user_transcript`, `agent_response`, `interruption`, `client_tool_call`, `vad_score`.
  - Formats: `pcm_8000`…`pcm_48000`, `ulaw_8000`. Pricing is about $0.08/min plus the LLM **(unverified)**.
- **Deepgram Voice Agent** [51]
  - A cascade configured with one `Settings` message: `listen` (Deepgram), `think` (OpenAI, Anthropic, Google, Groq, Bedrock or a custom URL) and `speak` (Deepgram, ElevenLabs, Cartesia, OpenAI, Polly).
  - Messages include `FunctionCallRequest`, `UserStartedSpeaking`, `AgentAudioDone`.
  - Audio: default 24 kHz; many codecs.
- **Speechmatics:** the voice-agent documentation now centres on "Agent STT" (`wss://global.rt.speechmatics.com/v2/agent`). The status of the older cascaded "Flow" agent is **unclear (unverified)** [52].

### 3.9 Cloud comparison table

| API | Protocol / transport | Audio in → out | Turn detection | Tools | List price | Session / context limits |
|---|---|---|---|---|---|---|
| OpenAI Realtime (gpt-realtime-2.x) | Realtime GA JSON; WS, WebRTC, SIP | PCM16 24 kHz (G.711 μ-law) → same | server_vad, semantic_vad, manual | functions, parallel via items; OOB responses | audio $32 / $64 per 1M (≈$0.019 / $0.077 per min, derived) | 60 min; 128k (2.x) or 32k (v1) |
| OpenAI Realtime mini (2.1-mini) | same | same | same | same | audio $10 / $20 per 1M | same |
| OpenAI GPT-Live-1 | **Live** session.* JSON; WS, WebRTC, SIP | not documented in sources | model-managed full duplex; mute | delegation (Responses or client) | $0.05/min + backend | 128k; `expired` close; 25–500 concurrent |
| Azure OpenAI Realtime | Realtime GA; WS, WebRTC, SIP | PCM16 24 kHz | same as OpenAI | same | Azure pricing | 2.x: no `truncation` |
| Azure Voice Live | Realtime-compatible superset; WS (+SDKs) | Azure STT/TTS or native | + Azure semantic VAD, EOU, noise / echo | functions (async), MCP | Pro / Basic / Lite tiers | 100k TPM default |
| Gemini Live (3.8) | BidiGenerateContent JSON; WSS (+ partner WebRTC) | PCM16 **16 kHz** → 24 kHz | automatic VAD (sensitivities) or manual activity | functions, non-blocking + scheduling, cancellation | audio $3 / $12 per 1M ($0.005 / $0.018 per min) | 15 min (audio) w/o compression; ~10 min connection; resume 2 h; 128k |
| Nova 2 Sonic | Bedrock HTTP/2 bidi event stream | LPCM **16 kHz** → 24 kHz | model-managed + endpointing sensitivity | toolUse / toolResult, async | ~$3 / $12 per 1M (unverified) | **8-min connection**; 1M context |
| xAI Grok Voice | Realtime-compatible WSS | PCM 8–48 kHz, G.711, Opus 24 kHz | server_vad or none | functions, web / X / file search, MCP | $0.08/min | resumption 30 min idle |
| Qwen-Omni-Realtime | Realtime-style (beta names); WS, WebRTC | PCM16 **16 kHz** → 24 kHz | server / semantic VAD, manual | functions, web search | n/a | 120 min |
| Hume EVI | proprietary WSS | linear16 → WAV (base64) | model-managed | yes | ~$0.04–0.07/min (unverified) | 30 min |
| Ultravox | proprietary (WebRTC / WS) | s16le PCM, any rate | model-managed | client / server tools | $0.05/min | 5 concurrent (PAYG) |

### 3.10 Independent benchmark snapshot

Artificial Analysis reports these columns [53]: Speech-to-Speech Index, Speech Reasoning (Big Bench Audio), Conversational Dynamics (Full-Duplex-Bench v1/v1.5), Agentic Performance, Arena Elo, Task Success, and Time to First Audio. The page is dynamic; these rows were quoted verbatim on 2026-09-24.

| Model | S2S Index | Speech Reasoning | Conv. Dynamics | TTFA (s) | $/h |
|---|---|---|---|---|---|
| Gemini 3.8 Live Extended Thinking (High) | 82.6 | 98% | 91.9% | 1.35 | 3.50 |
| GPT-Live-1 (Astra, medium) | 81.5 | 90% | 94.9% | 1.34 | 5.83 |
| Grok Voice Think Fast 2.0 High | 81.3 | 97% | 95.1% | **0.70** | 4.80 |
| GPT-Realtime-2 (High) | 73.6 | 97% | 95.3% | 1.14 | 4.14 |
| Nova 2.0 Sonic | n/a | 88% | n/a | 1.14 | 4.89 |
| Qwen3.5 Omni Plus Realtime | n/a | 99% | n/a | 2.64 | n/a |
| Step-Audio R1.1 (Realtime) | n/a | 98% | n/a | 1.53 | n/a |

Our own benchmark suite should reproduce TTFA and conversational dynamics locally, because these vary with region and network.

---

## 4. Open-weights models

### 4.1 Full-duplex family

- **Kyutai Moshi** [54][56][55]
  - Architecture: a 7B temporal transformer plus depth transformer over the **Mimi** codec (24 kHz audio → 12.5 Hz frames, 1.1 kbps, 80 ms frames), with an "inner monologue" text stream.
  - Latency: 160 ms theoretical, about 200 ms on an L4.
  - Hardware: PyTorch bf16 needs about **24 GB**. Backends are PyTorch (bf16, int8), MLX (int4, int8, bf16) and Rust/candle (int8, bf16).
  - License: weights CC-BY-4.0.
  - Limitation: **no tool access**.
  - Server: `python -m moshi.server`, `moshi_mlx.local -q 4`, or the Rust `moshi-backend`.
  - Wire protocol: WebSocket binary frames with a 1-byte type (0 handshake, 1 audio as Ogg/Opus 24 kHz mono, 2 text, 3 control, 4 metadata, 5 error, 6 ping).
  - Kyutai's 2026 work on top of Moshi includes MoshiRAG (asynchronous knowledge retrieval from a text LLM, 2026-04-30) [59], RL-tuned `moshika-rl-seamless` / `personaplex-rl-seamless` (2026-06-02) [60], and Hibiki-Zero speech translation (2026-02) [103].
- **NVIDIA PersonaPlex-7B-v1** (released 2026-01-15) [61][62][63]
  - Built on Moshiko weights, with Mimi at 24 kHz and a dual-stream design.
  - Conditioned on a **voice prompt plus a text role prompt**.
  - License: NVIDIA Open Model License, commercial use allowed.
  - FullDuplexBench: smooth turn-taking takeover rate 0.908 with 0.170 s latency; user-interruption takeover rate 0.950 with 0.240 s latency.
  - Tested on an A100 80 GB. Runs with `python -m moshi.server`, with `--cpu-offload` for small GPUs.
  - Community MLX 4/8-bit [64] and GGUF q4_k [102] ports exist. The runtime for those GGUFs is unclear, and the MLX ports carry a CC-BY-NC tag **(verify license)**.
- **MiniCPM-o 4.5** (2026-02) [72][73]
  - 9B, end to end: SigLIP2, Whisper-medium, CosyVoice2, Qwen3-8B.
  - Apache-2.0.
  - **Full-duplex, proactive** (time-division multiplexing, 1 Hz speak decisions); speech in English and Chinese with voice cloning.
  - VRAM: bf16 19 GB; **int4 11 GB**. The PyTorch full-duplex demo needs 28 GB or more; the llama.cpp path supports full duplex on GPUs with 12 GB or more.
  - Runtimes: llama.cpp, Ollama, vLLM, SGLang.
- **Realtime-Venus** (inclusionAI, 2026-09-16) [74]
  - 9B Omni and Audio checkpoints adapted from MiniCPM-o 4.5; Apache-2.0.
  - Full-duplex, and emits in-stream **`<delegate>`** requests that an async harness executes. This is the open analogue of GPT-Live delegation.
- **Delegation research:** a Sep 2026 paper describes a duplex speech-to-text front end that emits a delegation token and forwards ASR to a text LLM backend for tool calls. It reports 92–97% tool-call recall and beats GPT-realtime-mini and Qwen3-Omni on EVA-Bench [97].

### 4.2 Turn-based native S2S ("omni") models

- **Qwen3-Omni-30B-A3B (Instruct / Thinking)** [65][66]
  - Thinker–Talker MoE, Apache-2.0.
  - 234 ms theoretical first-packet latency.
  - Languages: 119 text, 19 speech input, 10 speech output.
  - VoiceBench overall 85.5 (Instruct) / 88.8 (Thinking), vs GPT-4o-Audio 86.8 and Gemini-2.5-Pro 89.6.
  - Minimum BF16 memory (transformers): **78.85 GB** (Instruct, 15 s video). Disabling the talker saves about 10 GB.
  - Serving: `vllm serve` covers only the thinker (text output) [65]. **vLLM-Omni** serves the full Thinker → Talker → Code2Wav pipeline behind `/v1/chat/completions` with `modalities` [68]. It reports 632–655 ms time to first audio at concurrency 64 on three GPUs [70]. OpenAI-Realtime alignment for Qwen3-Omni is still an RFC (2026-08-25) [69].
- **Qwen2.5-Omni-7B / 3B** [67]
  - Licenses: 7B Apache-2.0; 3B Qwen research license.
  - Minimum BF16 memory (15 s video): 31.11 GB (7B) and 18.38 GB (3B). Official AWQ and GPTQ-Int4 builds exist.
- **Step-Audio 2 mini** (8.3B, Apache-2.0) [77]: supports **tool calling** and multimodal RAG. **Step-Audio-R1.1** (2026-01) is an audio-reasoning, text-output model [94].
- **Kimi-Audio-7B-Instruct** (~9.8B total, MIT tag) [78]: hybrid continuous and discrete input, with a chunk-wise streaming flow-matching detokenizer.
- **GLM-4-Voice-9B** [79]: English and Chinese; Kyutai's 2026 RL "voice-of-reason" fine-tunes score 0.706 on GSM8K (written channel, GPT-4o judge) [80].
- **Fun-Audio-Chat-8B** (Tongyi, Dec 2025, Apache-2.0) [81]: 5 Hz / 25 Hz dual-resolution speech, speech function-calling benchmarks, about 24 GB for inference.
- **MiMo-Audio-7B-Instruct** (Xiaomi, MIT) [82].
- **LFM2.5-Audio-1.5B** (Liquid AI) [75][76][101]
  - Components: 1.2B LM plus a 115M FastConformer encoder and a Mimi-compatible 8-codebook detokenizer.
  - English only; LFM Open License v1.0.
  - Interleaved generation for S2S; llama.cpp GGUFs for CPU.
  - VoiceBench 54.92, vs Moshi 29.51, Mini-Omni2 33.49 and Qwen2.5-Omni-3B 63.57 on the same scale.
  - LFM2-Audio claimed under 100 ms end-to-end latency.
- **Research-grade models** (not recommended as defaults):
  - LLaMA-Omni 2 (0.5–32B; weights **non-commercial**) [83]
  - VITA-Audio (~8B, Apache-2.0) [84]
  - Mini-Omni2 (MIT) [85]
  - FlashLabs Chroma-4B (Apache-2.0, voice cloning, Mimi) [86]
  - LongCat-Flash-Omni (560B MoE / 27B active, MIT; needs 8×H20 or more) [87]
  - Freeze-Omni was not re-verified in this pass **(unverified)**.

### 4.3 Speech-in / text-out models and cascade components

- **Voxtral** (Mistral)
  - Small 24B and Mini 3B (Apache-2.0, audio → text) [91].
  - **Voxtral Mini 4B Realtime 2602** (Apache-2.0): streaming ASR with 80 ms–2.4 s configurable delay (480 ms recommended), 13 languages, served on **vLLM `/v1/realtime`** [90][71].
- **Gemma 4 E2B / E4B / 12B** (Apache-2.0): audio in, text out, native function calling [92]. llama.cpp supports E2B/E4B audio [95].
- **Nemotron 3 Nano Omni 30B-A3B** (2026-04-28): audio, video and image in, text out, tool calling. The NVFP4 weights are 21 GB, and the minimum listed GPU is an RTX 5090 32 GB [93].
- **Kyutai STT / TTS (delayed streams)** [58][57]
  - STT 1B en/fr with 0.5 s delay and semantic VAD; STT 2.6B en with 2.5 s delay.
  - `moshi-server` serves 64 concurrent streams on an L40S.
  - **Unmute** wraps STT, any OpenAI-compatible LLM and TTS behind a Realtime-like WebSocket. It needs 16 GB of VRAM or more and runs on Linux or WSL only.
- **Sesame CSM-1B** (Apache-2.0, gated) [88]: a conversational TTS conditioned on dialogue context, not a dialogue model.

### 4.4 Open models comparison

| Model | Params | License | Duplex | Audio I/O | VRAM (reported) | Latency (reported) | Tools | Serving |
|---|---|---|---|---|---|---|---|---|
| Moshi (moshiko / moshika) | 7B | CC-BY-4.0 | **Full** | 24 kHz Mimi / Opus | ~24 GB bf16; int8 / q4 / q8 | 160 ms theor., ~200 ms L4 | no | moshi.server, Rust, MLX |
| PersonaPlex-7B-v1 | 7B | NVIDIA OML (commercial) | **Full** | 24 kHz | A100-tested; CPU offload | FDB: 0.17 s turn-take, 0.24 s interrupt | no (role prompts) | moshi.server; community MLX / GGUF |
| MiniCPM-o 4.5 | 9B | Apache-2.0 | **Full** + proactive | EN / ZH speech | 19 GB bf16 / **11 GB int4** | n/a | n/a | transformers, llama.cpp, Ollama, vLLM |
| Realtime-Venus | 9B | Apache-2.0 | **Full** + delegate | EN / ZH | BF16 (size n/a) | n/a | async delegation | transformers (custom) |
| Qwen3-Omni-30B-A3B | 30B (3B active) | Apache-2.0 | Turn | multilingual (19 in / 10 out) | ≥68–79 GB BF16 | 234 ms theor. first packet | yes | transformers, vLLM (thinker), vLLM-Omni |
| Qwen2.5-Omni-7B / 3B | 7B / 3B | Apache-2.0 / research | Turn | 24 kHz out | 31 / 18 GB BF16 min; AWQ, GPTQ | n/a | n/a | transformers, vLLM, llama.cpp (in) |
| Step-Audio 2 mini | 8.3B | Apache-2.0 | Turn | EN / ZH | n/a | n/a | **yes** | transformers |
| Kimi-Audio-7B-Instruct | ~9.8B | MIT | Turn | EN / ZH | n/a | streaming detokenizer | n/a | kimi-audio |
| Fun-Audio-Chat-8B | ~8B | Apache-2.0 | Turn | EN / ZH | ~24 GB | n/a | yes (speech FC) | custom |
| GLM-4-Voice-9B | 9B | GLM-4-Voice license | Turn | EN / ZH | n/a | n/a | n/a | custom |
| LFM2.5-Audio-1.5B | 1.5B | LFM Open v1.0 | Turn (interleaved) | EN, 24 kHz out | small (CPU OK) | <100 ms claimed (v2) | n/a | liquid-audio, llama.cpp |
| Ultravox v0.7 (GLM-4.6) | adapter + GLM-4.6 | MIT | text out | ~50 languages in (per tags) | multi-GPU | n/a | via LLM | transformers, vLLM |
| Voxtral Mini 4B Realtime | 4B | Apache-2.0 | ASR only | 16 kHz in | on-device class | 80 ms–2.4 s delay | n/a | vLLM `/v1/realtime` |
| Gemma 4 E2B / E4B | 2.3B / 4.5B effective | Apache-2.0 | text out | audio in | small | n/a | yes | transformers, llama.cpp, MLX |

---

## 5. Protocol compatibility

### 5.1 OpenAI-Realtime-compatible endpoints

| Endpoint | Level | Differences to handle |
|---|---|---|
| OpenAI Realtime (GA) | reference | beta removed 2026-05-12 (changelog, via snippet) [1] |
| Azure OpenAI Realtime | same events [24] | URL, auth, `model=<deployment>`; 2.x rejects `truncation` |
| Azure Voice Live | "mostly match" + extensions [22] | extra VAD, noise and avatar fields; different endpoint and api-version |
| xAI Grok Voice Agent | drop-in by base URL [39] | cumulative `...transcription.updated`; no `conversation.item.done`; extra `force_message`, `replace`, resumption [40] |
| Alibaba Qwen-Omni-Realtime | Realtime-style [42] | **beta names** (`response.audio.delta` etc.); 16 kHz input; workspace URLs |
| vLLM `/v1/realtime` | "inspired by" [71] | ASR-oriented subset; PCM16 16 kHz; streaming models only |
| vLLM-Omni | experimental GA codec (MiniCPM duplex); Qwen3 profile proposed [69] | `?profile=openai-realtime` vs `qwen3-legacy` |
| Kyutai Unmute | "based on" + extensions [57] | cascade underneath |
| Speaches | spec + extensions [98] | cascade; **no `response.cancel` / `conversation.item.truncate`** |
| LocalAI | pipeline VAD+STT+LLM+TTS; WS + WebRTC [99] | extension events (`conversation.item.speaker`) |
| LiteLLM proxy | `/v1/realtime` routing for OpenAI, Azure, xAI, Gemini, Vertex, Bedrock [100] | translation depth for Gemini / Bedrock **(unverified)** |

**Not compatible** (need their own adapters): Gemini Live, Nova 2 Sonic, GPT-Live-1, Moshi / PersonaPlex, Hume, Ultravox, ElevenLabs, Deepgram.

### 5.2 Chat-completions endpoints with audio

- **OpenAI `gpt-audio`:** audio in and out on `/v1/chat/completions`, 128k context, $32 / $64 per 1M audio tokens [15].
- **Gemini OpenAI-compat:** `input_audio` input only [31].
- **DashScope Qwen-Omni compatible-mode:** `modalities: ["text","audio"]`, `audio: {voice, format}`, **`stream=True` required**, 24 kHz output [43].
- **vLLM-Omni:** Qwen3-Omni audio output via `modalities` [68].
- **llama.cpp `llama-server`:** OpenAI-compatible multimodal chat for Ultravox 0.5, Voxtral Mini, Qwen2.5-Omni, Qwen3-Omni and Gemma 4 E2B/E4B. Audio input; no audio output is documented [95].

A single "chat-audio" adapter plus library-side VAD therefore gives turn-based S2S over four stacks.

---

## 6. Implications for voice-agent-next

### 6.1 Which engines to build first

1. **`openai_realtime` (WebSocket first, WebRTC later) with compat profiles.**
   - Profiles: `openai`, `azure_openai`, `azure_voice_live`, `xai`, `qwen_omni`, `vllm`, `unmute` / `speaches` / `localai`.
   - This gives the highest coverage per line of code: the reference protocol, #1 market share, and our benchmark baseline.
2. **`gemini_live`**
   - The cheapest top-tier option ($0.005 / $0.018 per min) and a leaderboard leader.
   - It forces us to design GoAway, resumption, non-blocking tools and 16 kHz input correctly from day one.
3. **`moshi` (local, full-duplex)**
   - One binary-protocol adapter covers Moshi, PersonaPlex and Kyutai's RL / RAG variants on CUDA or MLX. It is our true full-duplex local baseline.
4. **`chat_audio`** (turn-based: gpt-audio, DashScope Qwen-Omni, vLLM-Omni) with our own VAD and turn manager. This is the path to Qwen-Omni locally or in the cloud.
5. **Phase 2:**
   - `nova_sonic` (Bedrock HTTP/2; 8-min rotation)
   - `gpt_live` (new Live protocol with delegation)
   - `local_omni` in-process runners for MiniCPM-o 4.5 / LFM2.5-Audio (llama.cpp or transformers)
6. **Phase 3:** hosted agent platforms (Ultravox, ElevenLabs, Deepgram, Hume) as "managed pipeline" engines.

### 6.2 Protocol-level abstraction

Model every engine as an **async bidirectional event stream** with a declared capability set. Keep it close to OpenAI GA semantics, because most backends map onto it, but add the concepts that other APIs expose.

```python
# sketch — voice_agent_next/engines/base.py
@dataclass(frozen=True)
class AudioFormat:            # carried on EVERY audio frame
    encoding: Literal["pcm16", "g711_ulaw", "g711_alaw", "opus"]
    sample_rate: int          # 8000 | 16000 | 24000 | 48000 ...
    channels: int = 1

@dataclass(frozen=True)
class EngineCapabilities:
    full_duplex: bool; server_vad: bool; semantic_vad: bool; manual_turns: bool
    client_truncation_required: bool      # OpenAI-WS yes, WebRTC/Gemini no
    tool_calls: Literal["none", "blocking", "non_blocking", "delegation"]
    tool_cancellation: bool; input_transcripts: bool; output_transcripts: bool
    text_input: bool; image_input: bool; reasoning_effort: bool
    max_session_s: int | None; max_connection_s: int | None; resumable: bool
    input_formats: tuple[AudioFormat, ...]; output_formats: tuple[AudioFormat, ...]

# commands (app -> engine)
Configure(instructions, voice, tools, turn_detection, reasoning_effort, modalities)
AppendAudio(frame: bytes, fmt: AudioFormat, t_capture_ms: int)
CommitInput() | ClearInput() | SendText(role, text) | InjectContext(text, speak: bool)
CreateResponse(overrides=None) | CancelResponse() | TruncateOutput(item_id, played_ms)
ToolResult(call_id, output, is_error=False) | SetMuted(bool) | Close()

# events (engine -> app)
SessionStarted | SessionUpdated | SpeechStarted(t_ms) | SpeechStopped(t_ms)
InputCommitted(item_id) | InputTranscript(item_id, text, final: bool)
ResponseStarted(response_id) | AudioDelta(pcm, fmt, item_id) | TextDelta(...)
OutputTranscript(item_id, text, final, phase: "commentary"|"final_answer"|None)
Interrupted(response_id, played_ms) | ResponseDone(status, usage)
ToolCall(call_id, name, args, mode) | ToolCallCancelled(call_ids)
DelegationStarted(id, target) | Usage(tokens_by_modality | seconds)
SessionExpiring(time_left_s) | ResumeHandle(handle) | Error(code, msg, fatal) | Closed(reason)
```

**Mapping of normalized events to provider events:**

| Normalized | OpenAI Realtime | Gemini Live | Nova 2 Sonic | GPT-Live-1 | Moshi |
|---|---|---|---|---|---|
| AppendAudio | `input_audio_buffer.append` | `realtimeInput.audio` | `audioInput` | `session.input_audio.append` | MT=1 Opus |
| CommitInput | `input_audio_buffer.commit` | `activityEnd` / `audioStreamEnd` | audio `contentEnd` | n/a (duplex) | n/a |
| InjectContext | `conversation.item.create` | `clientContent` | `textInput` | `session.thinking/commentary.append` | n/a |
| SpeechStarted | `input_audio_buffer.speech_started` | implicit / `interrupted` | in-band interrupted | transcripts | n/a |
| AudioDelta | `response.output_audio.delta` | `serverContent.modelTurn` | `audioOutput` | `session.output_audio.delta` | MT=1 |
| Transcripts | `…input_audio_transcription.*`, `response.output_audio_transcript.*` | `input/outputTranscription` | `textOutput` (role, stage) | `session.input/output_transcript.delta` | MT=2 text |
| Interrupted | speech_started while playing → truncate | `serverContent.interrupted` | `{ "interrupted" : true }` | overlap is native | n/a |
| ToolCall | `function_call` items | `toolCall` | `toolUse` + `contentEnd(TOOL)` | `response.event` / `session.delegation.created` | n/a |
| ToolResult | `function_call_output` + `response.create` | `toolResponse` | `toolResult` (TOOL content) | `response.item.create` + `response.create` | n/a |
| ToolCallCancelled | n/a | `toolCallCancellation` | n/a | n/a | n/a |
| ResponseDone | `response.done` | `turnComplete` / `generationComplete` | `contentEnd` / `completionEnd` | n/a | n/a |
| Usage | `response.done.usage` | `usageMetadata` | n/a | `session.usage.updated` | n/a |
| SessionExpiring | (60-min cap) | `goAway.timeLeft` | client timer (8 min) | `session.closed(expired)` | n/a |

Design rules:
- An **event-name alias table** handles beta-era and GA names (Qwen, vLLM) and xAI's cumulative transcripts.
- **Playback-position tracking** belongs to the audio output layer, not to the engines. It drives `TruncateOutput` wherever `client_truncation_required` is true.
- **Tool execution is always asynchronous** in our runtime. For blocking engines we simply await before replying. Cancellation propagates `ToolCallCancelled` or interruption to running tasks, because GPT-Live explicitly does *not* cancel backend work on barge-in [19].

### 6.3 Pitfalls checklist

1. **Resampling.** Use a single high-quality resampler (e.g. soxr), a canonical internal format of PCM16 mono at 24 kHz, and per-engine conversion: 16 kHz for Gemini, Nova and Qwen input; Opus/Ogg for Moshi; G.711 for telephony. Never infer the rate; carry it with each frame.
2. **Session and connection limits.**
   - Implement `SessionExpiring` handling with seamless rotation: Gemini resumption handles and context compression [27]; Nova's 8-min pattern [36]; OpenAI's 60-min hard cap [8].
   - Where no resumption exists (OpenAI, GPT-Live), re-seed from stored text transcripts, as GPT-Live's guide suggests [18].
   - Buffer user audio during handover.
3. **Truncation correctness.** Over WebSocket, report the *played* milliseconds, not the received ones. Include buffered output latency in the calculation [8].
4. **Context and cost growth.** Every response re-reads history. Use `retention_ratio`, cached-input pricing and summarization [10]. Note that Azure 2.x has no `truncation` field [24].
5. **Event drift.** GA vs beta names, xAI's missing events, and Speaches not supporting cancel or truncate [40][42][98].
6. **Voice immutability** after the first audio on OpenAI [8]. Native-audio Gemini returns audio only, so enable transcription for text [26].
7. **Echo.** Full-duplex local models, and any server VAD, will self-interrupt without acoustic echo cancellation. Browsers and Hume's SDK enable AEC [46]. Desktop Python needs our own AEC or a headset mode; Azure Voice Live offers server-side echo cancellation [22].
8. **Pacing in benchmarks.** Stream files at real-time pace (20–40 ms chunks). VAD and turn-taking behave differently on burst uploads. Measure TTFA from `SpeechStopped` (or end of file) to the first `AudioDelta`, matching Artificial Analysis' definition [53].
9. **Platform gaps.**
   - Unmute is Linux / WSL only [57]; PersonaPlex documents Linux [62].
   - The Nova Python SDK is experimental [38].
   - The Moshi protocol needs libopus [62].
   - Prefer llama.cpp-based local engines for native Windows.
10. **Licenses.** Check weights licenses before bundling any default local model: CC-BY (Moshi), NVIDIA OML (PersonaPlex), LFM Open, non-commercial LLaMA-Omni 2, Qwen research (2.5-Omni-3B).

### 6.4 Local serving recommendations

**16 GB VRAM NVIDIA GPU**

| Goal | Recommendation | Notes |
|---|---|---|
| True full-duplex demo | Moshi / PersonaPlex in int8 (PyTorch int8 or Rust / candle q8), or PersonaPlex `--cpu-offload` | bf16 needs ~24 GB [54][62]; int8 VRAM not measured **(unverified)** |
| Full-duplex omni with better intelligence | **MiniCPM-o 4.5 int4 via llama.cpp** | 11 GB int4; full duplex on 12 GB+ [73] |
| Smallest turn-based S2S | LFM2.5-Audio-1.5B | English only; llama.cpp / CPU capable [75] |
| Best quality within budget | Cascade: Kyutai STT 1B or Voxtral Mini 4B Realtime → 4–8B LLM → Kyutai TTS | Unmute's reference setup needs ≥16 GB [57][90] |
| Qwen-Omni | Qwen2.5-Omni-7B AWQ / GPTQ-Int4 (fit on 16 GB **unverified**) | Qwen3-Omni needs ≥68 GB; use cloud or multi-GPU [65][67] |

**Apple Silicon**
- Moshi via `moshi_mlx` q4/q8 (official) [54]; Kyutai STT MLX checkpoints [58].
- MiniCPM-o 4.5 via llama.cpp: M3/M4/M5 with 16 GB for half-duplex, M4 Max with 24 GB for full-duplex omni [73].
- LFM2.5-Audio GGUF [75]; Gemma 4 E2B/E4B audio-in via llama.cpp or MLX [92][95].
- mlx-audio for TTS / STT cascade parts (Voxtral Realtime, Kokoro, CSM) [96].
- PersonaPlex community MLX ports: check the license first [64].

### 6.5 Benchmark suite hooks

Emit the normalized events with monotonic timestamps so one harness can compute metrics for every engine:
- TTFA
- barge-in stop latency (`Interrupted` → last `AudioDelta`)
- transcript WER
- tool-call success
- $/min from `Usage`

Align task sets with public benchmarks where licenses allow:
- Big Bench Audio (speech reasoning)
- Full-Duplex-Bench v1/v1.5 (dynamics)
- VoiceBench
- ComplexFuncBench-Audio / EVA-Bench style tool tasks [53][33][97]

---

## 7. Open questions and unverified items

- Nova 2 Sonic, Hume, Deepgram and ElevenLabs list prices: third-party figures only.
- GPT-Live-1 audio formats, voices and latency: primary docs did not state formats; the community post gave numbers (§3.2).
- Azure's 256k vs OpenAI's 128k context for GPT-Realtime-2.x.
- Whether a Qwen3.5-Omni "Light" open-weights release exists: none found on the Qwen Hugging Face org.
- Real int8 / int4 VRAM for PersonaPlex. A MakeUseOf headline claims it runs on 8 GB **(unverified)**.
- LiteLLM's actual protocol translation for Gemini and Bedrock realtime.

---

## 8. Sources

*All accessed 2026-09-24 unless noted. "(snippet)" means seen via search result text only.*

1. OpenAI API Changelog — https://developers.openai.com/api/docs/changelog
2. GPT-Realtime model page — https://developers.openai.com/api/docs/models/gpt-realtime
3. GPT-Realtime-2 model page — https://developers.openai.com/api/docs/models/gpt-realtime-2
4. GPT-Realtime-2.1 model page — https://developers.openai.com/api/docs/models/gpt-realtime-2.1
5. GPT-Realtime-2.1 mini model page — https://developers.openai.com/api/docs/models/gpt-realtime-2.1-mini
6. GPT-Realtime mini model page — https://developers.openai.com/api/docs/models/gpt-realtime-mini
7. GPT-Realtime-1.5 model page — https://developers.openai.com/api/docs/models/gpt-realtime-1.5
8. OpenAI, Realtime conversations guide — https://developers.openai.com/api/docs/guides/realtime-conversations
9. OpenAI, Voice activity detection guide — https://developers.openai.com/api/docs/guides/realtime-vad
10. OpenAI, Managing Realtime costs — https://developers.openai.com/api/docs/guides/realtime-costs
11. OpenAI, Realtime WebRTC guide — https://developers.openai.com/api/docs/guides/realtime-webrtc
12. OpenAI, Realtime SIP guide — https://developers.openai.com/api/docs/guides/realtime-sip
13. OpenAI, Realtime transcription guide — https://developers.openai.com/api/docs/guides/realtime-transcription
14. OpenAI, Realtime client events reference (snippet) — https://developers.openai.com/api/reference/resources/realtime/client-events
15. GPT-Audio model page — https://developers.openai.com/api/docs/models/gpt-audio
16. GPT-Live 1 model page — https://developers.openai.com/api/docs/models/gpt-live-1
17. OpenAI, Getting started with GPT-Live — https://developers.openai.com/api/docs/guides/live
18. OpenAI, Managing GPT-Live sessions — https://developers.openai.com/api/docs/guides/live-conversations
19. OpenAI, Delegation and tools in GPT-Live — https://developers.openai.com/api/docs/guides/live-delegation
20. OpenAI Developer Community, "Introducing GPT-Live-1 in the API" (Sep 2026) — https://community.openai.com/t/introducing-gpt-live-1-in-the-api/1396471
21. Latent Space AINews, "GPT-Realtime-2, -Translate, and -Whisper" (May 2026) — https://www.latent.space/p/ainews-gpt-realtime-2-translate-and
22. Microsoft Learn, Voice Live API overview (updated 2026-09-16) — https://learn.microsoft.com/en-us/azure/ai-services/speech-service/voice-live
23. Microsoft Learn, Voice Live FAQ (updated 2026-06-05) — https://learn.microsoft.com/en-us/azure/ai-services/speech-service/voice-live-faq
24. Microsoft Learn, GPT Realtime 2.x overview (updated 2026-09-23) — https://learn.microsoft.com/en-us/azure/foundry/openai/concepts/realtime-2
25. Google, Gemini Live API overview — https://ai.google.dev/gemini-api/docs/live-api
26. Google, Live API capabilities guide — https://ai.google.dev/gemini-api/docs/live-api/capabilities
27. Google, Live API session management — https://ai.google.dev/gemini-api/docs/live-api/session-management
28. Google, Live API WebSockets reference — https://ai.google.dev/api/live
29. Google, Gemini API pricing — https://ai.google.dev/gemini-api/docs/pricing
30. Google, Gemini models — https://ai.google.dev/gemini-api/docs/models
31. Google, Gemini OpenAI compatibility — https://ai.google.dev/gemini-api/docs/openai
32. Google Blog, "Build real-time voice applications with Gemini 3.8 Live…" (2026-09-15) — https://blog.google/innovation-and-ai/technology/developers-tools/build-real-time-voice-applications-gemini-audio/
33. Google Blog, "Gemini 3.1 Flash Live" (2026-03-26) — https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-3-1-flash-live/
34. Google Cloud, Gemini Live API overview (Agent Platform) — https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/live-api
35. AWS, Nova 2 Sonic model card (Bedrock) — https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-amazon-nova-2-sonic.html
36. AWS, Getting started with speech-to-speech (Nova 2 Sonic) — https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-getting-started.html
37. AWS, Speech-to-Speech (Amazon Nova 2 Sonic) — https://docs.aws.amazon.com/nova/latest/nova2-userguide/using-conversational-speech.html
38. AWS, Using the Bidirectional Streaming API (Nova v1) — https://docs.aws.amazon.com/nova/latest/userguide/speech-bidirection.html
39. xAI, Speech to Speech / Voice Agent API — https://docs.x.ai/developers/model-capabilities/audio/voice-agent
40. xAI, Voice Agent guide — https://docs.x.ai/docs/guides/voice/agent
41. xAI, API pricing — https://docs.x.ai/developers/pricing
42. Alibaba Cloud Model Studio, Qwen-Omni-Realtime — https://www.alibabacloud.com/help/en/model-studio/realtime
43. Alibaba Cloud Model Studio, Qwen-Omni (OpenAI-compatible) — https://www.alibabacloud.com/help/en/model-studio/qwen-omni
44. MarkTechPost, "Alibaba Qwen Releases Qwen3.8-Omni-Flash" (2026-09-18) — https://www.marktechpost.com/2026/09/18/alibaba-qwen-releases-qwen3-8-omni-flash/
45. Hume, Speech-to-Speech (EVI) overview — https://dev.hume.ai/docs/speech-to-speech-evi/overview
46. Hume, EVI audio guide — https://dev.hume.ai/docs/speech-to-speech-evi/guides/audio
47. Ultravox, How Ultravox Works — https://docs.ultravox.ai/gettingstarted/how-ultravox-works
48. Ultravox, WebSocket Integration — https://docs.ultravox.ai/apps/websockets
49. Ultravox, Pricing — https://www.ultravox.ai/pricing
50. ElevenLabs, Agents WebSocket API reference — https://elevenlabs.io/docs/agents-platform/api-reference/agents-platform/websocket
51. Deepgram, Voice Agent API reference — https://developers.deepgram.com/reference/voice-agent/voice-agent
52. Speechmatics, Voice agents docs — https://docs.speechmatics.com/voice-agents-flow
53. Artificial Analysis, Speech-to-Speech leaderboard (snapshot 2026-09-24) — https://artificialanalysis.ai/speech-to-speech
54. kyutai-labs/moshi (GitHub) — https://github.com/kyutai-labs/moshi
55. Moshi streaming protocol (rust/protocol.md) — https://github.com/kyutai-labs/moshi/blob/main/rust/protocol.md
56. kyutai/moshiko-pytorch-bf16 model card — https://huggingface.co/kyutai/moshiko-pytorch-bf16
57. kyutai-labs/unmute (GitHub) — https://github.com/kyutai-labs/unmute
58. kyutai-labs/delayed-streams-modeling (GitHub) — https://github.com/kyutai-labs/delayed-streams-modeling
59. Kyutai blog, "MoshiRAG: Asynchronous Knowledge Retrieval for Full-Duplex Speech Language Models" (2026-04-30) — https://kyutai.org/blog/2026-04-30-moshi-rag/
60. kyutai/personaplex-rl-seamless (HF, 2026-06-02) — https://huggingface.co/kyutai/personaplex-rl-seamless
61. nvidia/personaplex-7b-v1 model card (2026-01-15) — https://huggingface.co/nvidia/personaplex-7b-v1
62. NVIDIA/personaplex (GitHub) — https://github.com/NVIDIA/personaplex
63. Roy et al., "PersonaPlex…", arXiv 2602.06053 (2026-01) — https://arxiv.org/abs/2602.06053
64. aufklarer/PersonaPlex-7B-MLX-4bit (community) — https://huggingface.co/aufklarer/PersonaPlex-7B-MLX-4bit
65. Qwen/Qwen3-Omni-30B-A3B-Instruct model card — https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct
66. Qwen3-Omni Technical Report, arXiv 2509.17765 — https://arxiv.org/abs/2509.17765
67. Qwen/Qwen2.5-Omni-7B model card — https://huggingface.co/Qwen/Qwen2.5-Omni-7B
68. vLLM-Omni, Qwen3-Omni online serving — https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/examples/online_serving/qwen3_omni/
69. vllm-omni issue #6592, "RFC: Align Qwen3-Omni Streaming Input with the OpenAI Realtime API" (2026-08-25) — https://github.com/vllm-project/vllm-omni/issues/6592
70. vLLM Blog, "Experience and Lessons Learned from Serving Multi-Stage Qwen3-Omni in vLLM-Omni" (2026-07-01) — https://vllm.ai/blog/2026-07-01-qwen3-omni-optimization
71. vLLM Blog, "Streaming Requests & Realtime API in vLLM" (2026-01-31) — https://vllm.ai/blog/2026-01-31-streaming-realtime
72. openbmb/MiniCPM-o-4_5 model card — https://huggingface.co/openbmb/MiniCPM-o-4_5
73. OpenBMB/MiniCPM-o (GitHub) — https://github.com/OpenBMB/MiniCPM-o
74. inclusionAI/Realtime-Venus (2026-09-16) — https://huggingface.co/inclusionAI/Realtime-Venus
75. LiquidAI/LFM2.5-Audio-1.5B model card — https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B
76. Liquid AI blog, "LFM2-Audio" (2025-10-01) — https://www.liquid.ai/blog/lfm2-audio-an-end-to-end-audio-foundation-model
77. stepfun-ai/Step-Audio-2-mini — https://huggingface.co/stepfun-ai/Step-Audio-2-mini
78. moonshotai/Kimi-Audio-7B-Instruct — https://huggingface.co/moonshotai/Kimi-Audio-7B-Instruct
79. zai-org/glm-4-voice-9b — https://huggingface.co/zai-org/glm-4-voice-9b
80. kyutai/glm-4-voice-of-reason-9b (2026-08) — https://huggingface.co/kyutai/glm-4-voice-of-reason-9b
81. FunAudioLLM/Fun-Audio-Chat-8B — https://huggingface.co/FunAudioLLM/Fun-Audio-Chat-8B
82. XiaomiMiMo/MiMo-Audio-7B-Instruct — https://huggingface.co/XiaomiMiMo/MiMo-Audio-7B-Instruct
83. ICTNLP/LLaMA-Omni2-7B — https://huggingface.co/ICTNLP/LLaMA-Omni2-7B
84. VITA-MLLM/VITA-Audio-Plus-Vanilla — https://huggingface.co/VITA-MLLM/VITA-Audio-Plus-Vanilla
85. gpt-omni/mini-omni2 — https://huggingface.co/gpt-omni/mini-omni2
86. FlashLabs/Chroma-4B — https://huggingface.co/FlashLabs/Chroma-4B
87. meituan-longcat/LongCat-Flash-Omni — https://huggingface.co/meituan-longcat/LongCat-Flash-Omni
88. sesame/csm-1b — https://huggingface.co/sesame/csm-1b
89. fixie-ai/ultravox-v0_7-glm-4_6 — https://huggingface.co/fixie-ai/ultravox-v0_7-glm-4_6
90. mistralai/Voxtral-Mini-4B-Realtime-2602 — https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602
91. mistralai/Voxtral-Small-24B-2507 — https://huggingface.co/mistralai/Voxtral-Small-24B-2507
92. google/gemma-4-E4B-it — https://huggingface.co/google/gemma-4-E4B-it
93. nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 (2026-04-28) — https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16
94. stepfun-ai/Step-Audio-R1.1 (2026-01) — https://huggingface.co/stepfun-ai/Step-Audio-R1.1
95. llama.cpp, multimodal docs — https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md
96. Blaizzy/mlx-audio (GitHub) — https://github.com/Blaizzy/mlx-audio
97. Hu et al., "A frontend-backend architecture for tool calls in full-duplex speech models", arXiv 2609.19334 (2026-09) — https://arxiv.org/abs/2609.19334
98. Speaches, Realtime API — https://speaches.ai/usage/realtime-api/
99. LocalAI, Realtime API — https://localai.io/docs/features/openai-realtime/
100. LiteLLM, /realtime docs — https://docs.litellm.ai/docs/realtime
101. Liquid4All/liquid-audio (GitHub) — https://github.com/Liquid4All/liquid-audio
102. Codes4Fun/personaplex-7b-v1-q4_k-GGUF (community) — https://huggingface.co/Codes4Fun/personaplex-7b-v1-q4_k-GGUF
103. kyutai/hibiki-zero-3b-pytorch-bf16 (HF, 2026-02-09) — https://huggingface.co/kyutai/hibiki-zero-3b-pytorch-bf16
