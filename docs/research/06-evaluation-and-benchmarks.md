# 06: Evaluation and Benchmarks for Voice Agents and Speech Models

> Research snapshot, **2026-09-24**, written as input for the `voice-agent-next` built-in benchmark suite.
> Numbers come from the cited sources. Anything not confirmed from a primary source is marked *(unverified)*.
> `[n]` refers to §9. Section 8 is a **proposal** for this project.

---

## 1. Executive summary

- **No single benchmark covers a voice agent.** Evaluation happens at five layers: components (ASR/TTS/VAD/turn detection/LLM), pipeline latency, conversational dynamics, spoken task success, and human preference.
  - The Artificial Analysis (AA) Speech-to-Speech Index, launched 2026-06-23, follows this pattern: Big Bench Audio (reasoning) + a Full-Duplex-Bench subset (dynamics) + τ-Voice (agentic).
  - It later added an arena and a task-success component [7][8][9].
- **Agentic voice is the hardest axis, and it is moving fast.**
  - τ-Voice (Mar 2026): voice agents solved 31–51% of tasks on clean audio and 26–38% on realistic audio, against ~85% for text agents [36]. Sierra reports the frontier rising from 30% (Aug 2025) to 67% (Apr 2026) [37].
  - EVA-Bench: no system scores above 0.5 on both accuracy and experience [41].
  - EchoChain: no system above 50% on keeping task state correct after an interruption. MP-Bench: ≤22% on multiparty comprehension [51][52].
- **Full-Duplex-Bench (FDB) is the standard full-duplex family.** v1 covers turn-taking, v1.5 overlap, v2 examiner-driven multi-turn, v3 tool use under real disfluency [10]–[14]. Newer 2026 sets (TurnBench, M3-DuplexBench, ECHO, EchoChain, MP-Bench) complement it [48]–[52][54].
- **Neither cascades nor native S2S win everywhere.**
  - Cascades lead on content: Whisper-v3-large + GPT-4o outranks GPT-4o-Audio on VoiceBench [4], and a GPT-4o cascade tops VocalBench [21].
  - Native S2S leads elsewhere: much better turn-taking on EVA-Bench (0.82–0.83 vs 0.28–0.58) [41], and GPT-Realtime beat a cascade on FDB-v3 tool use [13].
  - So both must run on identical stimuli, with quality *and* latency reported together.
- **"Latency" is not one metric.** TTFA, TTFB, TTFT, TTFS, turn latency and `e2e_latency` start and stop the clock at different points [7][74][99][104][111][112].
  - In one LiveKit report, adding up component metrics missed the real silence by 0.27–0.68 s [101].
  - Best practice: record the call and measure from end of user speech to start of agent speech [109][110][112].
- **Turn detection has a proper protocol.** LiveKit's eot-bench sweeps endpointing policies and reports two things across 14 languages (CC BY 4.0 data) [93]–[95]:
  - false-cutoff rate at fixed latency (300/600 ms);
  - latency at fixed false-cutoff (5/10%).
- **ASR evaluation is mature, but normalization is a trap.**
  - The Open ASR Leaderboard standardizes Whisper-style normalization, WER + RTFx, and fixed hardware. It added private Appen test sets in May 2026 to limit overfitting to public sets [65][66][69].
  - Voice agents also need streaming metrics: TTFS and "semantic WER" [74]–[76].
- **Objective TTS metrics are weak proxies.**
  - Common metrics: round-trip WER and speaker similarity (Seed-TTS-eval conventions) [77], plus MOS predictors (UTMOSv2, DNSMOS, NISQA).
  - TTSDS2 found these correlate inconsistently with human ratings across domains [79]. NISQA weights are non-commercial [83].
- **Simulation is the default for agent QA.** Harnesses combine an LLM-driven caller, TTS, noise/codec effects and deterministic end-state checks (τ-Voice, EVA) [37][41]. Commercial platforms (Coval, Hamming, Cekura, Bluejay, Roark) compute timing metrics from stereo recordings, e.g. stop-time-after-interruption, false barge-in and backchannel-yield [112][114][115].
- **Reusable OSS exists:** tau2-bench (MIT), ServiceNow EVA (MIT), livekit/eot-bench (Apache-2.0), pipecat-ai/stt-benchmark (BSD-2), coval-ai/benchmarks (Apache-2.0), open_asr_leaderboard (Apache-2.0) [39][42][93][75][88][66].
- **Recommendation (§8):** one stimulus-driven harness that measures at the audio boundary.
  - Seven tracks: latency, ASR, TTS, VAD/turn-taking, S2S quality, tool use, framework overhead.
  - Three tiers: smoke (CI, ≤10 min, CPU, no API keys), nightly, full.
  - Outputs: JSONL results plus stereo WAV files.
  - A strict reproducibility manifest (details in §8.6).

---

## 2. What to measure: a layered view

A 2025 taxonomy paper says to pick evaluations by three things: the *aspect* measured, the *capabilities* required (e.g. streaming generation), and the *protocol* needs [62]. Two findings matter here:

- **Static benchmarks predict preference poorly.** In Talk Arena's study (484 participants, 7,500 interactions), no single one of 20 static datasets correlated strongly with interactive preference (τ ≤ 0.33) [16].
- **Small subsets are good enough for regressions.** HUMANS shows that 50-item subsets reach >0.93 Pearson correlation with full-benchmark scores, but only ~0.85 with human satisfaction [60].

So cheap subsets can catch regressions, but quality claims still need interactive or preference checks.

| Layer | Question | Typical metrics | Representative benchmarks |
|---|---|---|---|
| Component | Accurate and fast? | WER/CER, RTFx, TTFS; TTFA, MOS-pred, round-trip WER, SIM; VAD AUC/F1; EOT false-cutoff | Open ASR LB, Seed-TTS-eval, FLEURS-VAD-102, eot-bench |
| Pipeline latency | How long is the silence? | voice-to-voice p50/p95, barge-in stop latency | Pipecat/LiveKit metrics, AA TTFA |
| Conversational dynamics | Talks and stops at the right time? | takeover rate, response/stop latency, false barge-in | FDB v1/v1.5/v2, TurnBench, ECHO |
| Spoken content | Right answer from speech? | accuracy, judge score, robustness delta | VoiceBench, Big Bench Audio, URO-Bench, VocalBench |
| Task success | Goal achieved? | pass@1, pass^k, tool F1 | τ-Voice, EVA-Bench, FDB-v3, VoiceAgentBench |
| Preference | Do people prefer it? | Elo / Bradley-Terry | Talk Arena, S2S-Arena, TTS Arena, AA arena |

---

## 3. Spoken-dialogue and speech-to-speech benchmarks

**Effort** in the tables below:
- **Low:** offline audio in, one scoring script, a few dollars of judge API.
- **Medium:** you write a per-model adapter, need a judge API, and download several GB.
- **High:** live real-time orchestration, several paid APIs, or human raters.

### 3.1 Half-duplex spoken QA and instruction following

| Benchmark | Measures | Size / format | HF id / hosting | License | Scoring | Effort |
|---|---|---|---|---|---|---|
| **VoiceBench** (2024) [1]–[4] | Knowledge, instruction following, safety, robustness to speaker/environment/content | 9 subsets (~20.6k rows): AlpacaEval-full 636, CommonEval 200, WildVoice 1,000, OpenBookQA 455, MMSU 3,074, SD-QA 553 × 11 accents, IFEval 345, BBH 1,000, AdvBench 520 | `hlt-lab/voicebench` | Apache-2.0 | MCQ exact match; IFEval rules; refusal rate; GPT-4o-mini judge for open-ended answers | Low |
| **Big Bench Audio** (2024) [5]–[7] | Spoken reasoning (4 BIG-Bench Hard categories) | 1,000 questions (4 × 250), 23 synthetic voices, ~293 MB | `ArtificialAnalysis/big_bench_audio` | MIT | LLM judge marks each answer correct/incorrect (originally Claude 3.5 Sonnet; Claude Sonnet 4.6 in AA v1.2). Since v1.1, non-answers count as wrong | Low |
| **URO-Bench** (2025) [17]–[19] | Understanding, reasoning, oral conversation; EN/ZH; multilingual, multi-round, paralinguistic; basic and pro tracks | 40 test sets; **URO-Bench-mini: 1,000 samples (25 per set)** | `Honggao/URO-Bench` | MIT | GPT-4o-mini judge + UTMOS + WER/CER of the spoken answer (Whisper-large-v3) | Medium |
| **VocalBench** (2025) [20]–[23] | Semantic, acoustic, chat, robustness (16 skills), plus RTF | 9,400 CosyVoice-synthesized items; zh and disfluency (DF) variants | `VocalNet/VocalBench`, `VocalNet/VocalBench-zh` | Apache-2.0 | Accuracy, UTMOS, WER, LLM judge (1–5), refusal/following rates, "preserve rate" under noise/reverb/far-field/packet loss/clipping | Medium |
| **SD-Eval** (2024) [26][27] | Replies that adapt to emotion, accent, age, background sound | 7,303 utterances, 8.76 h | `amphion/SD-Eval` | Data CC BY-NC 4.0; code Apache-2.0 | GPT-4o judge + reference metrics | Medium |
| **S2S-Arena** (2025) [24][25] | Paralinguistic instruction following, on input and output | Paper v2: 1,243 samples. HF release: 154 instructions / 21 tasks | `FreedomIntelligence/S2S-Arena` | CC BY-NC-SA 4.0 | Human pairwise arena | High |
| **VERA** (2025) [57] | Voice-native reasoning; gap between text and voice | 2,931 episodes, 5 tracks | GitHub `linyueqian/VERA` | *(unverified)* | Accuracy | Medium |

**Key results:**
- VoiceBench: the Whisper-v3-large + GPT-4o cascade scores 87.80, above GPT-4o-Audio at 86.75. Newer omni models lead at ~88–90 [4].
- VERA: the best text models average 54.0% across tracks; voice systems average 11.3% [57].
- S2S-Arena: cascades beat jointly trained models, and generating fitting paralinguistics is still hard [24].

**Also relevant:** WavBench (reasoning, colloquialism, paralinguistics) [58] and VoiceAssistant-Eval (10,497 examples, 13 categories) [59].

### 3.2 Audio-understanding suites (audio in, text out)

| Benchmark | Measures | Size | HF id | License | Scoring |
|---|---|---|---|---|---|
| **AudioBench** [28] | ASR, spoken QA, audio-scene QA, emotion/accent/gender, captioning | 50+ datasets | org `AudioLLMs` | per dataset *(unverified)* | WER, BLEU, METEOR, Llama-3-70B-Instruct judge |
| **AIR-Bench** [29][30] | Foundation tasks (19 tasks, ~19k multiple-choice) + chat (2k open questions) over speech, sound and music | ~21k items | `qyang1021/AIR-Bench-Dataset` | Code Apache-2.0; data card CC BY-NC 4.0 | Exact match; GPT-4 (`gpt-4-0125-preview`) judge |
| **MMAU / MMAU-Pro** [31][32] | Expert-level audio reasoning | MMAU: 10k clips / 27 tasks (test-mini 1k public, test 9k with hidden answers). Pro: 5,305 items / 49 skills | `gamma-lab-umd/MMAU-test-mini`, `gamma-lab-umd/MMAU-Pro` | Cards CC BY-NC 4.0; code Apache-2.0 | Multiple-choice accuracy; hidden test scored via an HF Space |

These matter only if we ship audio-LLM engines.

### 3.3 Full-duplex and turn-taking benchmarks

| Version | Tasks and data | Metrics | Effort |
|---|---|---|---|
| **FDB v1** [10][14] | Pause handling (Candor 216 + synthetic 137), backchannel (ICC 55), smooth turn-taking (Candor 119), user interruption (synthetic 200). Offline: the system writes `output.wav`, which is aligned with parakeet-tdt-0.6b-v2 | Takeover rate (TOR); response latency; backchannel frequency and JSD vs human timing; GPT-4o relevance 0–5. A backchannel is <1 s and <2 words; pauses are 0.4–1.0 s | Medium |
| **FDB v1.5** [11] | TTS-built overlap: user interruption 200, backchannel 99, talking to others 100, background speech 100 | Behaviour class (respond/resume/uncertain/unknown/silent). **Stop latency**: overlap onset → model stops. **Response latency**: overlap end → next utterance. Prosody + UTMOSv2 | Medium |
| **FDB v2** [12][14] | Live WebRTC Node.js orchestrator with an automated "Examiner" at fast/slow pacing. Topics: daily, correction, entity tracking, safety (ACL 2026) | LLM judge + latency | High |
| **FDB v3** [13][14] | 100 **real** recordings, 12 speakers, 4 domains, 5 disfluency types. Mock APIs answer instantly and deterministically. Data on Google Drive | Tool-selection F1, argument accuracy (GPT-4o), Pass@1, turn-take rate, interruption rate, latency breakdown | Medium |

- **Licensing:** the FDB paper licenses are CC BY-NC-SA 4.0 (v1) and CC BY-SA 4.0 (v1.5, v3). Dataset licenses: *(unverified)*.
- **FDB-v3 results:**
  - Pass@1: GPT-Realtime 0.600, a Whisper → GPT-4o → TTS cascade 0.450, Ultravox v0.7 0.410.
  - The cascade had the slowest first word (8.78 s).
  - Self-correction was the hardest case; the cascade scored 0.176 there [13].

**Other turn-taking benchmarks:**
- **Talking Turns** (ICLR 2025): judges turn-taking with a model trained on human–human conversations [55].
- **FD-Bench:** 293 simulated conversations with 1,200 interruptions [56].
- **TurnBench** (2026): 30 h of hand-labelled dyadic speech in 6 interaction styles, 14 systems tested [54].
- **M3-DuplexBench:** English/Japanese [48].
- **ECHO:** matched pairs where the same words are either an interruption or a backchannel. Most systems are biased toward yielding the floor [50].
- **EchoChain:** 40.2% more failures in interrupted full-duplex runs than in half-duplex runs [51].
- **MP-Bench:** multiparty turn-taking is near chance [52].
- Model reports now cite these benchmarks. For example, Qwen-Audio-3.1-Realtime reports its FDB-v1.5 background-speech response rate falling from 73.0% to 13.0% [63].

### 3.4 Agentic / tool-use voice benchmarks

| Benchmark | Measures | Size | HF id / hosting | License | Scoring | Effort |
|---|---|---|---|---|---|---|
| **τ-Voice** (2026) [36]–[40] | Full-duplex customer-service tasks under realistic audio | 278 tasks (airline 50, retail 114, telecom 114) | `sierra-research/tau2-bench` (`--audio-native`) | MIT (code) | Final database state; pass@1 / pass^k (AA averages 3 trials) | High: provider adapters + ElevenLabs/Deepgram keys |
| **EVA-Bench** (2026) [41]–[43] | EVA-A (task completion, faithfulness, speech fidelity) + EVA-X (progression, conciseness, turn-taking) | 213 scenarios, 3 domains, bot-to-bot audio | `ServiceNow-AI/eva`; GitHub `ServiceNow/eva` | MIT | Database hash + LLM/audio-LM judges + timestamp turn-taking; pass@1, pass@k, pass^k | Medium-high (Pipecat-based) |
| **VoiceAgentBench** (2025) [34][35] | Single, parallel and sequential tool calls, multi-turn, safety; English + 6 Indic languages | 5,394 rows (paper: 6,000+) | `krutrim-ai-labs/VoiceAgentBench` | Krutrim Community License 1.0 | Parameter-filling accuracy + LLM judge | Medium |
| **Audio MultiChallenge** (2025) [44] | Memory, instruction retention, self-coherence, mid-utterance edits | 452 real conversations, 47 speakers, 1,712 rubrics | *(unverified)* | CC BY 4.0 per arXiv listing (data *unverified*) | Rubric pass rate (best 54.65%) | Medium |
| **SpokenWOZ** (2023) [33] | Human–human spoken task-oriented dialogue | 5.7k dialogues, 249 h, 8 domains | project site | CC BY-NC 4.0 | Joint goal accuracy, task success | Medium |
| **Voice Readiness / aiewf-eval** (2026) [106][108][122] | Text LLMs in 30-turn voice-agent scripts | 46 configurations | `kwindla/aiewf-eval` | *(unverified)* | A turn passes only if tool use, instruction following and knowledge-base grounding all pass; also TTFAT | Low |

**Other 2026 benchmarks:**
- **Audio2Tool:** ~30k spoken commands for car/home/wearables, CC BY 4.0 [47].
- **VAmoS Bench:** 100 scenarios with a seeded PostgreSQL backend. Binary assertions catch agents that *claim* a database change but never make it [45].
- **τ-Elicitation:** 200 entity-capture tasks; success 0.14–0.41. Reading entities back improves pass³ by 14–31 points but costs 21–28 s per call [53].
- **DuplexWorld:** 156 scenarios scored on agentic, conversational and naturalness (DNSMOS) axes [49].
- **"From Text to Voice":** converts text tool benchmarks (Confetti, When2Call) to audio. The text-to-voice gap is 1.8–4.8 points, and open Qwen3 judges (≥8B) agree with proprietary judges >80% of the time [46].

### 3.5 Human-preference arenas

- **Talk Arena** [15][16], **S2S-Arena** [24], and **AA's Speech Agent Arena**, added in index v1.1 [7].
- **TTS Arena V2:** blind pairwise voting, accounts ≥30 days old, English only, prompts ≤1,000 characters [78].
- Arenas are useful for checking automatic metrics, but they cannot run in CI.

---

## 4. Component benchmarks

### 4.1 ASR

**Open ASR Leaderboard method** [65][66][67]:
- **Tracks:** English short-form (≤30 s ESB test sets), multilingual (de/fr/it/es/pt), and long-form.
- **Normalization:** WER is computed after Whisper-style normalization (`EnglishTextNormalizer`, `BasicMultilingualTextNormalizer`). It handles numbers and spelling and removes fillers.
- **Speed:** **RTFx = total audio duration ÷ total transcription time**, on one fixed setup. The paper used A100-80GB at batch 64; by July 2026 the repo runs on HF Jobs (H200).
- **API models:** evaluated through the same open scripts.
- **Private track (May 2026):** Appen sets report scripted vs conversational WER and US vs non-US accent WER [69].

| Dataset | Domain | License | Access / HF id |
|---|---|---|---|
| LibriSpeech [71] | Read audiobooks | CC BY 4.0 | Open; `openslr/librispeech_asr` or ESB copy in `hf-audio/open-asr-leaderboard` |
| FLEURS [70] | Read speech, 102 languages, same sentences in every language | CC BY 4.0 | Open; `google/fleurs` |
| Common Voice | Read, crowd-sourced | CC0 | From v23 (2025-09-30) only via Mozilla Data Collective [72]; the ESB copy is gated [68] |
| Earnings-22 / AMI / VoxPopuli [68] | Earnings calls / meetings / parliament | CC BY-SA 4.0 / CC BY 4.0 / CC0 | Open; ESB configs |
| TED-LIUM; GigaSpeech; SPGISpeech [68] | Talks; podcasts; finance | CC BY-NC-ND 3.0; Apache-2.0 (gated); Kensho agreement (gated) | ESB configs |
| Pipecat STT set [74] | 1,000 short conversational turns | Not stated on card | `pipecat-ai/stt-benchmark-data` |

**Pitfalls:**
- `BasicTextNormalizer` can corrupt Indic scripts; use `preserve_marks=True` [73].
- Use CER for zh/ja/th-style scripts [18][77].
- "Semantic WER" counts only errors that would change what an LLM agent does, as judged by an LLM (Claude) [74]. It is useful but non-deterministic, so report it next to classic WER, not instead of it.

**Streaming metrics:**
- **TTFS** (Daily/Pipecat) = time from the VAD stop event to the final transcript. The benchmark streams 20 ms chunks at 16 kHz in real time from one US site [74].
- Speechmatics calls TTFB "the worst metric in the industry" because it can be gamed. It recommends partial latency, finals latency and TTFS instead [76].
- **Results:** Pipecat's Pareto frontier is Deepgram Nova 3 (247 ms / 1.62% WER), Soniox (249 ms / 1.29%) and Speechmatics (495 ms / 1.07%). P95 latency differs ~5× between providers [74].

### 4.2 TTS

| Metric | Definition | Implementation | Caveat |
|---|---|---|---|
| TTFA | Request → first *audible* sample. Leading silence counts | Pipecat TTFA [104]; Coval TTFA p50–p99 and P25–P75 spread [88][89] | Silence padding can make TTFB look better than it is |
| Generation time / RTF | Wall time for ~500 characters, or synthesis time ÷ audio duration | AA: 4 runs/day at random times, 14-day median, download time included [87]; Coval [88] | Streaming engines also need underrun checks |
| Round-trip WER/CER | ASR of the synthesized audio vs the input text | Seed-TTS-eval: Whisper-large-v3 (en), Paraformer-zh (zh) [77]; Coval [89] | Depends on the ASR and normalizer |
| Speaker similarity | Cosine similarity of speaker-verification embeddings | WavLM-large fine-tuned for speaker verification [77] | Voice cloning only |
| MOS predictors | Predicted quality without a reference. Options: UTMOSv2 (MIT; VoiceMOS 2024 winner, trained on synthetic speech) [82]; DNSMOS P.835 SIG/BAK/OVRL [84][85]; NISQA (code MIT, **weights CC BY-NC-SA**) [83]. VERSA bundles 65 metrics [86] | — | TTSDS2: its own metric was the only one of 16 above 0.50 Spearman in every domain [79] |
| Hard text | Audio-LM judge on emotion, foreign words, URLs, formulas, questions | EmergentTTS-Eval, 1,645 cases [80][81] | Judges have biases [61] |

Seed-TTS-eval [77] has three sets: test-en (1,000 Common Voice samples), test-zh (2,000 DiDiSpeech-2 samples), and a hard subset. It is distributed via Google Drive with no license stated.

### 4.3 VAD

- **Silero** reports ROC-AUC over 31.25 ms chunks on a multi-domain mix: ESC-50, AliMeeting, Earnings21, MSDWild, AISHELL-4, VoxConverse, LibriParty and private calls. v6 scores 0.97; WebRTC VAD scores 0.73 [90].
- **FLEURS-VAD-102** (FireRedVAD): 9,443 hand-labelled files in 102 languages; release "coming soon". The vendor-run table reports AUC / F1 / false-alarm rate / miss rate [91]:
  - Silero: 97.99 / 95.95 / 9.41 / 3.95.
  - TEN VAD: 97.81 / 95.19 / 15.47 / 2.95.
- **TEN VAD** publishes precision-recall curves with 10/16 ms hops. It reports RTF of 0.005–0.057 across devices and claims faster speech-to-silence transitions. License: Apache-2.0 with extra conditions [92].
- **Gap:** nobody standardizes onset/offset *latency*, which is what matters for endpointing.

### 4.4 End-of-turn detection

**LiveKit eot-bench** [93]–[95]:
- **Method:** every pause inside a user turn is a decision point. The harness sweeps each model's endpointing policy and reports:
  - false-cutoff rate at 300 and 600 ms latency budgets;
  - dead-air latency at 5% and 10% false cutoffs;
  - the full Pareto curve.
  - A VAD-only baseline is included.
- **Data:** `livekit/eot-bench-data`, CC BY 4.0. ~5.5k turns in 14 languages (≤400 per language, seed 20260603). The harness has adapters for LiveKit, SmartTurn, UltraVAD, VAP, Deepgram, AssemblyAI, Cartesia and others (Apache-2.0).
- **Results:**

  | Latency budget | False-cutoff rate |
  |---|---|
  | 300 ms | LiveKit v1 9.9%, Deepgram Flux 12.9%, ultraVAD 27.7% |
  | 600 ms | LiveKit v1 4.5%, Soniox 5.5% |

**Smart Turn v3.2** [96]–[98]:
- ~8M parameters on a Whisper-Tiny base; up to 8 s of 16 kHz audio; BSD-2.
- v3.1 accuracy (8 MB / 32 MB models): 94.7 / 95.6% English, 90.1 / 91.0% Spanish.
- CPU inference: 9–73 ms per call.
- Test sets have 31.5k rows; no license is stated on the dataset card.

**Terminology:** a *false cutoff* (ending the user's turn too early) is not the same as LiveKit's "false interruption", where user audio interrupts the agent but yields no words [102].

### 4.5 LLM for voice

- **TTFAT** (time to first *answer* token): measured to the first user-visible or tool-call token, excluding separately streamed reasoning [106].
- **Voice Readiness** [105][106]:
  - Re-ranks 46 LLM configurations within a ~700 ms LLM budget, assuming ~500 ms for the rest of the pipeline.
  - Reports p50/p95.
  - Labels local RTX 5090 runs as "over localhost with no network latency".
- **Daily's earlier LLM benchmark** used a Claude Opus judge. For S2S models it measured voice-to-voice latency from waveforms with Silero VAD, against a <1,500 ms target [108].
- **PhoneBench** calibrates a panel of LLM judges against human labels and reports cost per minute [107].

---

## 5. Measuring latency and turn-taking in practice

### 5.1 Where the clock starts and stops

| Source | Metric | Start | Stop |
|---|---|---|---|
| AA [7][9] | TTFA | Request with question audio (exact start point not specified) | First audio token (mean over Big Bench Audio) |
| Daily primer [109]; Modal [110] | Voice-to-voice | End of user speech *in a recording* | Start of bot speech (Modal used pyannote diarization) |
| Pipecat [104] | `UserBotLatencyObserver` | User stops speaking | Bot starts speaking |
| LiveKit [99][100] | `e2e_latency` ≈ EOU delay + `llm.ttft` + `tts.ttfb` | VAD end of speech | First TTS byte (Realtime models: first audio token) |
| Hamming [111] | Turn latency; TTFW | User silence ends; call connects | Agent audio starts; first audio byte |
| Cekura [112][114] | Latency; stop time; TTS overrun | Caller finishes (stereo VAD); user speech onset | Agent starts; agent stops |
| Roark [115] | Turn-onset latency; barge-in stop latency (T90) | User end of utterance; user speech onset | First agent audio; TTS cut, "measured from audio, not from event logs" |
| FDB v1.5 [11] | Stop latency; response latency | Overlap onset; overlap end | Model stops; next utterance |

### 5.2 Budgets and observed numbers

- **Daily/Pipecat primer budget** [109]: ~1.29 s total.
  - AI processing: 300 ms transcription and endpointing, 650 ms LLM TTFB, 20 ms sentence aggregation, 120 ms TTS TTFB.
  - Transport, each direction: Opus encoding ~21 ms, 40 ms jitter buffer, 10 ms network, plus device I/O.
  - Target: 1,500 ms voice-to-voice. Typical human response: ~500 ms.
- **Modal** reached a ~1 s median, but only with the client and GPUs close together [110].
- **Hamming production data** (vendor-reported; method *unverified*): p50 1.4–1.7 s, p90 3.3–3.8 s, p95 4.3–5.4 s, p99 8.4–15.3 s [111].

### 5.3 Pitfalls

- **Component sums lie.** In a LiveKit user report, the sum predicted 3.98 s where 3.3 s was measured, and 2.33 s where 2.6 s was measured. The reporter suspected tool execution and sentence tokenization; the issue was closed without a maintainer answer [101]. LiveKit also reports `on_user_turn_completed_delay` (time spent in the user callback) as a separate field [99].
- **Leading silence** must count toward TTFA [88][104].
- **Location matters.** Pipecat tells users to measure from their own network location [74]. AA's TTS timing includes download time [87].
- **Averages hide failures:** report P95/P99 [113][117].
- **Load changes behaviour:** test at 2× peak concurrency on real transports [116].

### 5.4 Turn-taking vocabulary

| Metric | Definition | Source |
|---|---|---|
| Takeover rate (TOR) | Share of pause/backchannel events where the system takes the floor | [10] |
| Backchannel JSD | Divergence from human backchannel timing, in 200 ms bins | [10] |
| False barge-in rate | Share of agent stops not caused by a real interruption (cough, keyboard, TV, second voice) | [115] |
| Backchannel yield rate | Share of "mhmm"/"right" events after which the agent abandons its turn | [115] |
| Talk-over duration | ms of overlap per turn boundary, attributed to whoever started it | [115] |
| Stop time after interruption | Cekura counts <500 ms as responsive | [112] |
| Background-speech response rate | Share of non-addressed speech the agent answers | [11][63] |

---

## 6. Simulation-based evaluation and regression testing

**Simulators:**
- **τ-Voice** [37][40]:
  - GPT-4.1 writes the caller's turns and ElevenLabs speaks them in a chosen persona.
  - The audio is degraded with noise, vocal tics, G.711 μ-law at 8 kHz, muffling, and Gilbert–Elliott frame drops.
  - Every 2 s an LLM policy decides whether to interrupt, yield or backchannel; the orchestrator advances in 200 ms ticks.
  - Runs are seeded and replayable.
  - Stated limits: the simulated caller reads the agent's transcript instead of hearing it through ASR; speech quality is not scored; accents come from TTS [37].
- **EVA** [41][42]:
  - The caller is itself a cascade (Scribe-v2.2-Realtime + GPT-5.1 + ElevenLabs v3) and talks to the agent over WebSocket.
  - Judges check each simulated conversation for fidelity and regenerate the failures. 12.0% of trials needed regeneration.
- **FDB v2** uses an automated Examiner [12].
- **SpokenUS** (EMNLP 2026) is a *trained* simulator that barges in, is disfluent, and has a turn-taking head [64].
- **Vendors:**
  - Coval: personas and test sets, plus GitHub Actions integration with Pipecat [118].
  - Bluejay: 500+ scenarios, and a "golden" set of 50 conversations re-run on every change [117].
  - Roark: replays past production failures as regression tests [115][116].

**Scoring, from most to least deterministic:**
1. Final database state (τ-Voice; EVA's state hash) [37][41].
2. Binary assertions over call traces [45].
3. Per-instance rubrics [44].
4. Pinned LLM judges. AA swapped its judge between methodology versions [7].
5. Audio-LM judges: up to 0.91 Spearman with human preference, but with verbosity and position biases [61].

**Reliability:**
- pass@1 = average performance; pass@k = peak performance; pass^k = passes all k trials, i.e. consistency.
- EVA reports a median gap of 0.44 between peak and consistent performance [41].

**CI practice:** LiveKit recommends text-mode pytest/Vitest unit tests for single-turn behaviour and separate simulations for multi-turn [103]. Vendors re-run simulations on every change [117][118].

---

## 7. Reusable open-source tooling

| Tool | Covers | License | Use in voice-agent-next |
|---|---|---|---|
| `sierra-research/tau2-bench` [39][40] | τ-Voice full-duplex agentic eval | MIT | Implement its `DiscreteTimeAdapter` for our session |
| `ServiceNow/eva` [42][43] | EVA metrics, 213 scenarios, bot-to-bot audio | MIT | Connect to its WebSocket caller (effort *unverified*) |
| `DanielLin94144/Full-Duplex-Bench` [14] | FDB v1–v3 inputs and scorers | *(unverified)* | Offline v1/v1.5 replay |
| `livekit/eot-bench` [93] | End-of-turn policy sweep + adapters | Apache-2.0 | Add adapters for our turn detectors |
| `pipecat-ai/stt-benchmark` [75] | TTFS + semantic WER, 20+ STT vendors | BSD-2 | Reuse the TTFS method and dataset |
| `huggingface/open_asr_leaderboard` [66] | Dataset loaders, normalizer, RTFx | Apache-2.0 | Conventions for the ASR track |
| `whisper_normalizer` [73] | Whisper and Indic normalizers | MIT | Lightweight dependency |
| `coval-ai/benchmarks` [88] | STT/TTS latency and WER harness | Apache-2.0 | Audio hashing and loudness conventions |
| VoiceBench [2], URO-Bench [18], VocalBench [21] | S2S content scoring | Apache-2.0 / MIT / Apache-2.0 | Loaders and scorers for track T5 |
| UTMOSv2 [82], DNSMOS [85], NISQA [83] | MOS prediction | MIT / CC BY 4.0 (repo) / MIT code + NC weights | UTMOSv2 + DNSMOS by default; NISQA opt-in |
| VERSA [86], UltraEval-Audio [119], AU-Harness [120], Kimi-Audio-Evalkit [121] | Speech-metric and audio-LLM harnesses | AU-Harness Apache-2.0; others *(unverified)* | Optional backends |

---

## 8. Implications for voice-agent-next: proposed benchmark suite

### 8.1 Principles

1. **The recorded audio is the ground truth; traces explain it.** Every user-perceived number comes from the stereo recording. Internal spans only show where the time went, which avoids the component-sum error [101].
2. **One harness for every engine type.** Local and cloud cascades, native S2S and full-duplex engines all go through the same `Session` interface and transports.
3. **Fixed scripted stimuli first**, LLM-driven simulated callers second.
4. **Report distributions, not means**, and always show quality next to latency as a Pareto view [95][106].
5. **Tiered cost:** smoke on every PR, nightly, full per release.
6. **Interoperable:**
   - metric names map to Pipecat and LiveKit metrics [99][104];
   - adapters run τ-Voice, EVA, FDB and eot-bench unchanged.

### 8.2 Harness

- **CallerEmulator:**
  - Streams pre-rendered PCM stimuli in real-time 20 ms chunks [74].
  - Transports: in-process loopback, WebSocket, WebRTC, telephony (G.711, 8 kHz).
  - Records user (left) and agent (right) on one monotonic clock.
- **Stimulus manifest:** for every stimulus, the speech on/offsets, pause spans and interruption points. These are computed once on the clean audio and stored with it.
- **TraceCollector:** records framework events: `vad_start/stop`, `eou_decision`, `stt_final`, `llm_first_token`, `llm_first_answer_token`, `tool_start/end`, `tts_first_audio`, `audio_out_first_frame`, `interrupt`, `flush`.
- **Mocks:** scripted STT/LLM/TTS with configurable latency, plus recorded provider streams ("cassettes") that can be replayed. This makes CI deterministic and keyless.
- **Judges:** pinned by model id, prompt hash and temperature, with an open-weight option [46].

### 8.3 Tracks and metric definitions

**Notation.** Times are in ms on the harness clock.
- `t_uoff`: annotated end of user speech.
- `t_aon`: agent onset. This is the first 10 ms frame that begins at least 100 ms of speech on the agent channel, according to a reference VAD (Silero, p ≥ 0.5). Comfort noise and clicks do not count.

**T1 Latency**
- `v2v_ms` = `t_aon − t_uoff`, for every turn that expects a reply. Report n, mean, p50/p90/p95/p99 and max.
- Report separately: the first turn (cold start) and `session_ready_ms`.
- Span breakdown: `eou_delay`, `stt_ttfs`, `llm_ttft`, `llm_ttfat`, `tool_ms`, `tts_ttfa`, `transport_ms`.
- `residual_ms` = `v2v − Σspans`. This is framework time plus buffering.
- `dead_air_rate` = share of turns with `v2v` > 2,000 ms (Cekura's "good" threshold; configurable) [112].

**T2 ASR**
- `wer` = Σ(S+D+I) / ΣN over the whole corpus, after the Whisper English normalizer. Other languages use the basic normalizer with `preserve_marks`; zh/ja/ko/th use `cer`. Also report the per-utterance mean and % perfect [65][73][74].
- `entity_er`: errors on digits, emails and IDs.
- `rtfx`: batch throughput.
- Streaming: `ttfs_ms`, `first_partial_ms`, `finals_lag_ms` [74][76].
- `semantic_wer`: optional, marked as judge-based.

**T3 TTS**
- `ttfa_ms` (leading silence counts), `rtf`, and `underruns_per_min` (buffer starvation during real-time playback).
- `rt_wer`/`rt_cer` using a fixed ASR [77].
- `utmosv2` and `dnsmos_{sig,bak,ovrl}`.
- `spk_sim` for cloning engines [77].
- `hardtext_acc`: after round-trip ASR, are numbers, dates, emails and URLs read correctly? [80]
- MOS predictors are regression signals, not rankings [79].

**T4 VAD and turn-taking**
- **VAD:** frame-level ROC-AUC, F1, false-alarm rate and miss rate at 10 ms resolution [90][91]. Also `onset_lag_ms`/`offset_lag_ms` and CPU RTF.
- **Offline end-of-turn, following eot-bench:** `false_cutoff@300/600ms` and `latency@5/10%fc` per language [93]. Plus accuracy, precision, recall, F1 and false-positive rate on the Smart Turn test set.
- **Online, measured on the call recording:**
  - `premature_response_rate`: the agent starts talking during a 0.4–1.0 s mid-turn pause [10].
  - `missed_turn_rate`: no agent reply within 5 s.
  - `barge_in_stop_ms` (agent stops − interruption starts) and `stop_within_500ms_rate` [11][112][115].
  - `post_interrupt_response_ms` [11].
  - `false_barge_in_rate`, measured with a battery of noise, side-talk and background-speech clips [11][115].
  - `backchannel_yield_rate`: the agent pauses >300 ms after "mm-hmm" [115].
  - `overlap_ms_per_turn`, attributed to whoever started the overlap [115].

**T5 S2S quality**
- Datasets: Big Bench Audio accuracy [7], VoiceBench subsets [1], URO-Bench-mini [19].
- **Score the audio, not the engine's own text.** Transcribe the agent's audio with a fixed ASR, then judge it. `speech_fidelity` = how well the engine's text (if exposed) matches its audio [41].
- `voice_gap` = the same LLM's text-mode score − its voice-mode score [36][57].
- `robustness_delta` = clean score − score on degraded audio (VoiceBench variants; noise and G.711 as in τ-Voice) [1][37].

**T6 Tool use**
- `pass@1` (mean over k trials) and `pass^k` with k = 3 [41].
- `tool_f1` and `arg_acc` [13].
- `say_do_violation_rate`: the agent says it did something but made no matching tool call or state change [45].
- `entity_capture_acc`: names, emails and IDs in the final state are exactly right [53].

**T7 Framework overhead**
- Mock services with fixed delays, including a zero-delay variant: `overhead_ms` = `v2v − Σ injected delays` (p50/p99).
- `flush_ms`: interrupt decision → last agent audio frame leaves the transport.
- `frame_jitter_ms`: standard deviation of the gaps between output audio frames.
- Event-loop lag p99, CPU% and RSS per session.
- `sessions_per_core` while p95 overhead stays ≤50 ms (proposed target). Run every transport.

### 8.4 Tiers and datasets

| Track | Smoke (per PR; CPU; no keys; ≤10 min) | Nightly | Full / release |
|---|---|---|---|
| T1 | 20 scripted turns over loopback with mocks; 5 turns with small local models | 3 sessions × 40 turns per engine and transport; cloud from a fixed region | + 1/10/50 concurrent sessions; ≥2 regions; different times of day |
| T2 | LibriSpeech test-clean 50 + FLEURS 5 languages × 10 | 200 each from LS-clean/other, AMI, Earnings-22, VoxPopuli; FLEURS 10 × 100; Pipecat STT set 200 | Full ESB English + FLEURS; Common Voice if the user downloads it from MDC |
| T3 | 30 agent sentences we write ourselves (CC0) + 10 from Seed-TTS test-en | 200 test-en + 200 EmergentTTS + 100 agent sentences | All Seed-TTS sets + EmergentTTS 1,645 + human A/B spot checks |
| T4 | eot-bench `en` 100 turns; Smart Turn test 200; 2 min VAD clip; FDB v1 20 items | eot-bench, all 14 languages (~5.5k); FDB v1 + v1.5 (~1.2k items) | + FDB v2 live; noise and backchannel batteries |
| T5 | Big Bench Audio 20 (5 per category) from recorded replays | BBA 200; VoiceBench openbookqa 455, ifeval 345, advbench 520, sd-qa-usa 553, commoneval 200; URO-Bench-mini 1,000 | BBA 1,000; all of VoiceBench; VocalBench |
| T6 | 10 scripted scenarios with mock tools (text + pre-rendered audio) | One EVA domain (k=3) or a τ-Voice retail subset | τ-Voice 278 (k=3); EVA 213 (k=5); FDB-v3 100 |
| T7 | Mocks over loopback + WebSocket, 200 turns | + WebRTC, CPU profiles | + telephony; 1 h soak |

- Smoke results are **regression canaries**. Compare them with the stored baseline for the same kind of CI runner, and never publish them as capability scores.
- Nightly subsets are fixed lists of item IDs. The HUMANS method is one way to choose them [60].

### 8.5 Output format

`results/<run_id>/` contains:
- `manifest.json`
- `items.jsonl` (one line per item × trial)
- `summary.json` (n, mean, p50/p90/p95/p99, bootstrap 95% CI)
- `report.md` (optional HTML) and `junit.xml`
- `artifacts/<item>/{stereo.wav, labels.txt, trace.json, transcript.jsonl}`, matching τ-Voice and EVA output layouts [40][42]

Illustrative `summary.json` entry:

```json
{"suite":"van-bench","suite_version":"0.1.0","track":"latency","engine":"cascade:parakeet+llama+kokoro",
 "transport":"webrtc","dataset":"scripted-turns@sha256:…","n":120,
 "v2v_ms":{"p50":812,"p90":1040,"p95":1122,"p99":1390,"ci95_p50":[790,833]},
 "spans_p50_ms":{"eou_delay":210,"stt_ttfs":95,"llm_ttfat":310,"tts_ttfa":120,"residual":77},
 "first_turn_v2v_ms":1650,"dead_air_rate":0.008}
```

**CI gate:** fail a PR if p50 `v2v_ms` or `overhead_ms` gets worse by more than 10% **and** more than 30 ms with non-overlapping CIs, or if a smoke quality metric moves beyond its stored tolerance.

### 8.6 Reproducibility rules

1. **Manifest:**
   - suite version and git SHA;
   - engine config; model ids with HF revision SHAs; provider model/API versions; quantization;
   - CPU/GPU, driver/CUDA, RAM, OS/kernel; Python and lockfile hash;
   - transport; provider region with measured RTT and TLS handshake time;
   - dataset revisions with a SHA-256 per audio file [88];
   - judge id, prompt hash and temperature.
2. **Stimuli:** fixed resampler; loudness normalized to −20 dBFS RMS [88]; 20 ms real-time chunks [74]; fixed seeds.
3. **Warm-up:** discard the first session (or first 3 turns) for each engine. Report cold start separately.
4. **Repetitions:**
   - Latency: ≥100 turns per condition, over ≥3 separate sessions.
   - Nondeterministic quality metrics: k ≥ 3 trials. AA averages 3 [7]; EVA uses 5 [41].
5. **Statistics:** percentiles with bootstrap CIs, never means alone [111][117].
6. **Cloud timing:**
   - Randomize the order in which engines run.
   - Published comparisons sample several times of day. AA's TTS benchmark runs 4×/day and reports a 14-day median [87]; Coval samples every 30 min [88].
7. **Isolation:** run one engine at a time outside load tests. Record the CPU governor, GPU clocks and thermal state.
8. **Judge changes:** version them and re-check against a 50–100-item human-labelled set [107].
9. **Methodology changes:** bump the suite's major version and log them in a changelog, as AA does [7].

### 8.7 Fair comparisons

**Local vs cloud**
- Measure both at the same point in the audio path, with the same stimuli and transport. Headline number: user-perceived `v2v_ms`.
- Also report `network_ms` (RTT and handshake) and `compute_ms` (spans), so readers can see where the time goes.
- State the network topology: where the client, agent and provider each run.
  - Co-location dominates results; Modal's ~1 s median needed nearby containers [110].
  - Label localhost runs explicitly, as Voice Readiness does [106].
- Report cost per minute (API list price vs amortized hardware, as PhoneBench does [107]).
- Report capacity at a p95 target and error or rate-limit rates under load [116].

**Native S2S vs cascaded**
- Use the same stimuli and judges. Score the transcribed output audio plus `speech_fidelity` [41].
- Each system keeps its own endpointing: server-side VAD for S2S APIs, our VAD + turn detector for cascades. Add an ablation where the harness itself decides when the turn ends, if the provider allows it *(varies by provider; unverified)*.
- Publish per-dimension scores and a joint Pareto chart (quality vs p50 `v2v`). The winner changes by dimension:
  - content: cascades lead [4][21][24];
  - turn-taking and accents: S2S leads [41];
  - tool use: mixed [13][34].
- Always include a text-only baseline of the same LLM [36][57].
- Run the full-duplex tasks (FDB v1.5/v2) on any system that supports barge-in; label half-duplex systems as such.
- Equalize tool latency with deterministic mocks, as FDB-v3 does [13].

### 8.8 Data and licensing hygiene

- **Bundle as smoke subsets, with attribution files:** LibriSpeech and FLEURS (CC BY 4.0), VoxPopuli (CC0), AMI (CC BY 4.0), Big Bench Audio (MIT), VoiceBench (Apache-2.0), URO-Bench (MIT), VocalBench (Apache-2.0), eot-bench data (CC BY 4.0).
- **Download at run time only, never redistribute:** SD-Eval, AIR-Bench data, MMAU/MMAU-Pro, SpokenWOZ, S2S-Arena, TED-LIUM, NISQA weights, VoiceAgentBench, gated GigaSpeech/SPGISpeech.
- **No license stated; ask the maintainers first:** Pipecat STT and Smart Turn test sets, Seed-TTS-eval.
- **Common Voice** requires accepting the Mozilla Data Collective terms [72].

### 8.9 Roadmap and open questions

**Roadmap:**
1. Harness + T1/T7 with mocks; eot-bench adapter; ASR smoke tests; JSON output and CI gate.
2. T3/T5 and cloud engines; nightly runs; HTML report.
3. T6 via the τ-Voice adapter and EVA bridge; FDB v1/v1.5 replay; load tests.
4. FDB v2/v3 live; multiple regions; periodic human A/B checks of the automatic metrics.

**Open questions:**
- Licenses for Full-Duplex-Bench, the Smart Turn test sets and Seed-TTS-eval.
- Which VAD to use as the reference for detecting speech onsets. It slightly biases latency, so publish the choice and make it configurable.
- Budget for judge API calls in nightly runs.
- Whether to run a public leaderboard, possibly with a submit-by-PR-then-verify model like Open ASR's [65].

---

## 9. Sources

1. VoiceBench: Benchmarking LLM-Based Voice Assistants (arXiv 2410.17196, Oct 2024). https://arxiv.org/abs/2410.17196
2. VoiceBench GitHub repository. https://github.com/MatthewCYM/VoiceBench
3. `hlt-lab/voicebench` dataset card (HF, updated Apr 2025). https://huggingface.co/datasets/hlt-lab/voicebench
4. VoiceBench leaderboard page. https://matthewcym.github.io/VoiceBench/
5. Evaluating Audio Reasoning with Big Bench Audio (HF blog, 2024-12-20). https://huggingface.co/blog/big-bench-audio-release
6. `ArtificialAnalysis/big_bench_audio` dataset card (HF). https://huggingface.co/datasets/ArtificialAnalysis/big_bench_audio
7. Artificial Analysis, Speech-to-Speech Benchmarking Methodology (v1.0–v2.0, 2026). https://artificialanalysis.ai/methodology/speech-to-speech-benchmarking
8. Announcing the Artificial Analysis Speech to Speech Index (2026-06-23). https://artificialanalysis.ai/articles/announcing-the-artificial-analysis-speech-to-speech-index
9. Artificial Analysis Speech to Speech leaderboard. https://artificialanalysis.ai/speech-to-speech
10. Full-Duplex-Bench v1 (arXiv 2503.04721, Mar 2025). https://arxiv.org/abs/2503.04721
11. Full-Duplex-Bench v1.5: Evaluating Overlap Handling (arXiv 2507.23159, Jul 2025). https://arxiv.org/abs/2507.23159
12. Full-Duplex-Bench-v2 (arXiv 2510.07838, Oct 2025, rev. Apr 2026). https://arxiv.org/abs/2510.07838
13. Full-Duplex-Bench-v3: Tool Use Under Real-World Disfluency (arXiv 2604.04847, Apr 2026). https://arxiv.org/abs/2604.04847
14. Full-Duplex-Bench GitHub repository. https://github.com/DanielLin94144/Full-Duplex-Bench
15. Talk Arena blog. https://talkarena.org/blog
16. Mind the Gap! Static and Interactive Evaluations of Large Audio Models (arXiv 2502.15919, Feb 2025). https://arxiv.org/abs/2502.15919
17. URO-Bench (arXiv 2502.17810, Feb 2025). https://arxiv.org/abs/2502.17810
18. URO-Bench GitHub repository. https://github.com/Ruiqi-Yan/URO-Bench
19. `Honggao/URO-Bench` dataset card (HF). https://huggingface.co/datasets/Honggao/URO-Bench
20. VocalBench (arXiv 2505.15727, May 2025). https://arxiv.org/abs/2505.15727
21. VocalBench GitHub repository (leaderboard, updates to 2026-01). https://github.com/SJTU-OmniAgent/VocalBench
22. `VocalNet/VocalBench` dataset card (HF). https://huggingface.co/datasets/VocalNet/VocalBench
23. VocalBench-DF: Speech LLM Robustness to Disfluency (arXiv 2510.15406, Oct 2025). https://arxiv.org/abs/2510.15406
24. S2S-Arena (arXiv 2503.05085). https://arxiv.org/abs/2503.05085
25. `FreedomIntelligence/S2S-Arena` dataset card (HF). https://huggingface.co/datasets/FreedomIntelligence/S2S-Arena
26. SD-Eval GitHub repository. https://github.com/amphionspace/SD-Eval
27. `amphion/SD-Eval` dataset card (HF). https://huggingface.co/datasets/amphion/SD-Eval
28. AudioBench GitHub repository. https://github.com/AudioLLMs/AudioBench
29. AIR-Bench GitHub repository. https://github.com/OFA-Sys/AIR-Bench
30. `qyang1021/AIR-Bench-Dataset` dataset card (HF). https://huggingface.co/datasets/qyang1021/AIR-Bench-Dataset
31. MMAU GitHub repository. https://github.com/Sakshi113/MMAU
32. `gamma-lab-umd/MMAU-Pro` dataset card (HF, Aug 2025). https://huggingface.co/datasets/gamma-lab-umd/MMAU-Pro
33. SpokenWOZ (arXiv 2305.13040). https://arxiv.org/abs/2305.13040
34. VoiceAgentBench (arXiv 2510.07978, Oct 2025). https://arxiv.org/abs/2510.07978
35. `krutrim-ai-labs/VoiceAgentBench` dataset card (HF, updated Feb 2026). https://huggingface.co/datasets/krutrim-ai-labs/VoiceAgentBench
36. τ-Voice: Benchmarking Full-Duplex Voice Agents on Real-World Domains (arXiv 2603.13686, 2026-03-14). https://arxiv.org/abs/2603.13686
37. Sierra, "τ-voice: benchmarking real-time voice agents" (2026-05-01). https://sierra.ai/blog/tau-voice-benchmarking-real-time-voice-agents-on-real-world-tasks
38. Sierra, "τ³-Bench: advancing agent evaluation to knowledge and voice" (2026-03-18). https://sierra.ai/blog/bench-advancing-agent-benchmarking-to-knowledge-and-voice
39. `sierra-research/tau2-bench` GitHub repository (v1.0.1, Jul 2026). https://github.com/sierra-research/tau2-bench
40. tau2-bench voice full-duplex README. https://github.com/sierra-research/tau2-bench/blob/main/src/tau2/voice/README.md
41. EVA-Bench: A New End-to-end Framework for Evaluating Voice Agents (arXiv 2605.13841, May 2026, rev. Sep 2026). https://arxiv.org/abs/2605.13841 (HTML: https://arxiv.org/html/2605.13841v1)
42. `ServiceNow/eva` GitHub repository. https://github.com/ServiceNow/eva
43. `ServiceNow-AI/eva` dataset card (HF, Mar 2026). https://huggingface.co/datasets/ServiceNow-AI/eva
44. Audio MultiChallenge (arXiv 2512.14865, 2025-12-16). https://arxiv.org/abs/2512.14865
45. VAmoS Bench: Voice Agent Simulation Bench (arXiv 2607.27453, 2026-07-29). https://arxiv.org/abs/2607.27453
46. From Text to Voice: A Reproducible and Verifiable Framework for Evaluating Tool Calling LLM Agents (arXiv 2605.15104, May 2026). https://arxiv.org/abs/2605.15104
47. Audio2Tool: A Dataset for Benchmarking Speech Tool Use (arXiv 2604.22821, Apr 2026). https://arxiv.org/abs/2604.22821
48. M3-DuplexBench (arXiv 2607.29125, 2026-07-31). https://arxiv.org/abs/2607.29125
49. DuplexWorld (arXiv 2608.10716, 2026-08-11). https://arxiv.org/abs/2608.10716
50. ECHO: Matched-Contrast Benchmark for Context-Sensitive Turn-Taking (arXiv 2609.17360, 2026-09-15). https://arxiv.org/abs/2609.17360
51. EchoChain (arXiv 2604.16456, 2026-04-08). https://arxiv.org/abs/2604.16456
52. MP-Bench (arXiv 2609.13076, 2026-09-11). https://arxiv.org/abs/2609.13076
53. τ-Elicitation (arXiv 2609.13602, 2026-09-11). https://arxiv.org/abs/2609.13602
54. TurnBench (arXiv 2608.25218, 2026-08-25). https://arxiv.org/abs/2608.25218
55. Talking Turns (arXiv 2503.01174, ICLR 2025). https://arxiv.org/abs/2503.01174
56. FD-Bench (arXiv 2507.19040, Jul 2025). https://arxiv.org/abs/2507.19040
57. VERA: Voice Evaluation of Reasoning Ability (arXiv 2509.26542, 2025-09-30). https://arxiv.org/abs/2509.26542
58. WavBench (arXiv 2602.12135, Feb 2026). https://arxiv.org/abs/2602.12135
59. VoiceAssistant-Eval (arXiv 2509.22651, Sep 2025). https://arxiv.org/abs/2509.22651
60. Putting HUMANS first: Efficient LAM Evaluation with Human Preference Alignment (arXiv 2605.00022, ACL 2026). https://arxiv.org/abs/2605.00022
61. AudioJudge (arXiv 2507.12705, Jul 2025). https://arxiv.org/abs/2507.12705
62. Which Evaluation for Which Model? A Taxonomy for Speech Model Assessment (arXiv 2510.19509). https://arxiv.org/abs/2510.19509
63. Qwen-Audio-3.1-Realtime (arXiv 2609.25176, 2026-09-21). https://arxiv.org/abs/2609.25176
64. SpokenUS: A Spoken User Simulator for Task-Oriented Dialogue (arXiv 2603.16783, 2026). https://arxiv.org/abs/2603.16783
65. Open ASR Leaderboard paper (arXiv 2510.06961, v4 2026-03-30). https://arxiv.org/abs/2510.06961
66. `huggingface/open_asr_leaderboard` GitHub repository. https://github.com/huggingface/open_asr_leaderboard
67. `hf-audio/open-asr-leaderboard` dataset (formerly esb-datasets-test-only-sorted). https://huggingface.co/datasets/hf-audio/open-asr-leaderboard
68. ESB datasets card (per-dataset licenses). https://huggingface.co/datasets/esb/datasets
69. Appen press release: private data for the Open ASR Leaderboard (2026-05-06). https://www.appen.com/press-release/appen-hugging-face-open-asr-leaderboard
70. `google/fleurs` dataset card. https://huggingface.co/datasets/google/fleurs
71. `openslr/librispeech_asr` dataset card. https://huggingface.co/datasets/openslr/librispeech_asr
72. Common Voice 23.0 live on Mozilla Data Collective (2025-09-30). https://community.mozilladatacollective.com/common-voice-23-0-live-on-mozilla-data-collective/
73. `whisper_normalizer` GitHub repository. https://github.com/kurianbenoy/whisper_normalizer
74. Daily, "Benchmarking STT for Voice Agents" (2026-02-13). https://www.daily.co/blog/benchmarking-stt-for-voice-agents/
75. `pipecat-ai/stt-benchmark` GitHub repository. https://github.com/pipecat-ai/stt-benchmark
76. Speechmatics, "Speed you can trust: the STT metrics that matter for voice agents" (2026-04-14). https://www.speechmatics.com/company/articles-and-news/speed-you-can-trust-the-stt-metrics-that-matter-for-voice-agents
77. `BytedanceSpeech/seed-tts-eval` GitHub repository. https://github.com/BytedanceSpeech/seed-tts-eval
78. TTS Arena documentation. https://docs.ttsarena.org/
79. TTSDS2 (arXiv 2506.19441, Jun 2025, rev. Mar 2026). https://arxiv.org/abs/2506.19441
80. EmergentTTS-Eval (arXiv 2505.23009, NeurIPS 2025). https://arxiv.org/abs/2505.23009
81. `bosonai/EmergentTTS-Eval` dataset card (HF). https://huggingface.co/datasets/bosonai/EmergentTTS-Eval
82. `sarulab-speech/UTMOSv2` GitHub repository. https://github.com/sarulab-speech/UTMOSv2
83. `gabrielmittag/NISQA` GitHub repository. https://github.com/gabrielmittag/NISQA
84. DNSMOS P.835 (arXiv 2110.01763). https://arxiv.org/abs/2110.01763
85. `microsoft/DNS-Challenge` GitHub repository (DNSMOS; license per repo LICENSE, seen via search). https://github.com/microsoft/DNS-Challenge
86. VERSA: A Versatile Evaluation Toolkit for Speech, Audio, and Music (arXiv 2412.17667). https://arxiv.org/abs/2412.17667
87. Artificial Analysis, Text to Speech Benchmarking Methodology. https://artificialanalysis.ai/text-to-speech/methodology
88. `coval-ai/benchmarks` GitHub repository. https://github.com/coval-ai/benchmarks
89. Coval, "Benchmarks" (results retrieved 2026-05-13). https://www.coval.ai/blog/benchmarks
90. Silero VAD wiki: Quality Metrics. https://github.com/snakers4/silero-vad/wiki/Quality-Metrics
91. `FireRedTeam/FireRedVAD` GitHub repository. https://github.com/FireRedTeam/FireRedVAD
92. `TEN-framework/ten-vad` model card (HF). https://huggingface.co/TEN-framework/ten-vad
93. `livekit/eot-bench` GitHub repository. https://github.com/livekit/eot-bench
94. `livekit/eot-bench-data` dataset card (HF, Jul 2026). https://huggingface.co/datasets/livekit/eot-bench-data
95. LiveKit, "Solving end-of-turn detection: LiveKit Turn Detector v1.0" (2026-06-17). https://livekit.com/blog/solving-end-of-turn-detection
96. Daily, "Improved accuracy in Smart Turn v3.1" (2025-12-03). https://www.daily.co/blog/improved-accuracy-in-smart-turn-v3-1/
97. `pipecat-ai/smart-turn` GitHub repository. https://github.com/pipecat-ai/smart-turn
98. `pipecat-ai/smart-turn-data-v3.2-test` dataset card (HF, Jan 2026). https://huggingface.co/datasets/pipecat-ai/smart-turn-data-v3.2-test
99. LiveKit docs, "Capturing metrics". https://docs.livekit.io/agents/ops/logging/
100. LiveKit, "Understand and improve voice agent latency" (2026-04-13). https://livekit.com/blog/understand-and-improve-agent-latency
101. livekit/agents issue #3824, "How to measure latency properly?". https://github.com/livekit/agents/issues/3824
102. LiveKit docs, "Turns overview" (turn detection, interruptions, false interruptions). https://docs.livekit.io/agents/build/turns/
103. LiveKit docs, "Testing and evaluation". https://docs.livekit.io/agents/start/testing/
104. Pipecat docs, "Metrics". https://docs.pipecat.ai/guides/fundamentals/metrics
105. Pipecat Benchmarks index. https://www.pipecat.ai/benchmarks
106. Pipecat, "Voice Readiness" benchmark (2026-08-18). https://www.pipecat.ai/benchmarks/voice-readiness
107. Pipecat, "PhoneBench Alpha 1" (2026-08-27). https://www.pipecat.ai/benchmarks/phonebench-alpha-1
108. Daily, "Benchmarking LLMs for Voice Agent Use Cases" (2026-02-02). https://www.daily.co/blog/benchmarking-llms-for-voice-agent-use-cases/
109. Voice AI & Voice Agents: An Illustrated Primer (Feb 2025, updated Jun 2026). https://voiceaiandvoiceagents.com/
110. Modal, "One-second voice-to-voice latency with Modal, Pipecat, and open models" (2025-11-04). https://modal.com/blog/low-latency-voice-bot
111. Hamming, "Voice Agent Evaluation Metrics: Definitions, Formulas & Benchmarks" (2026-01-18). https://hamming.ai/resources/voice-agent-evaluation-metrics-guide
112. Cekura docs, "Pre-defined metrics". https://docs.cekura.ai/documentation/key-concepts/metrics/pre-defined-metrics
113. Cekura, "Voice AI Latency: How to Measure and Monitor Every Layer" (2026-09-03). https://www.cekura.ai/blogs/voice-ai-latency-guide
114. Cekura, "Barge-In: End-to-End Interruption Metrics Across ASR & TTS". https://www.cekura.ai/discover/voice-ai-barge-in-testing-asr-latency-tts-overrun
115. Roark, "Testing full-duplex voice agents after GPT-Live" (2026-07-13). https://roark.ai/blog/testing-full-duplex-voice-agents-gpt-live
116. Roark, "Load testing voice agents: the launch check most teams skip" (2026-08-27). https://roark.ai/blog/load-testing-voice-agents-before-launch
117. Bluejay, "Voice Agent Testing: The Complete Guide for 2026" (2026-03-09). https://getbluejay.ai/resources/voice-agent-testing-guide
118. Pipecat docs, "Evaluations: Coval". https://docs.pipecat.ai/pipecat/fundamentals/evaluations/coval
119. UltraEval-Audio (arXiv 2601.01373; ACL 2026 demo). https://arxiv.org/abs/2601.01373
120. AU-Harness (arXiv 2509.08031, Sep 2025). https://arxiv.org/abs/2509.08031
121. `MoonshotAI/Kimi-Audio-Evalkit` GitHub repository (released 2025-04-25, via search). https://github.com/MoonshotAI/Kimi-Audio-Evalkit
122. `kwindla/aiewf-eval` GitHub repository (linked from [106][108]; not read directly). https://github.com/kwindla/aiewf-eval
