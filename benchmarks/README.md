# voice-agent-next benchmarks (`van bench`)

One stimulus-driven harness for **every** engine — native speech-to-speech or cascade,
local or cloud — that measures what the user *hears*. Design rationale:
[research note 06, §8](../docs/research/06-evaluation-and-benchmarks.md) and
[note 05, §2.4](../docs/research/05-latency-transports-production.md).

| Track | What | Status |
| --- | --- | --- |
| **T1 latency** | voice-to-voice latency on the call recording, cold start, dead air | `van bench latency` |
| **T2 ASR** | WER/CER, RTFx, final latency (TTFS), first partial, interim stability | `van bench asr` |
| T3 TTS | TTFA, RTF, round-trip WER, MOS predictors | planned (#42) |
| **T4 VAD / turn-taking** | VAD frame metrics and onset/offset lag; end-of-turn on eot-bench; turn-taking battery (premature replies, barge-in, false barge-ins) | `van bench vad`, `van bench turns`, `van bench turn-taking` |
| T5 S2S quality | Big Bench Audio, VoiceBench | planned (#44) |
| T6 tool use | scripted tool scenarios, τ-Voice | planned (#45) |
| **T7 framework overhead** | `v2v − Σ injected delays`, flush, jitter, loop lag, capacity, hot paths; CI regression gate | `van bench overhead` |

## Quick start

```bash
# smoke tier: mock engine, 20 scripted turns, offline, < 1 min
van bench latency --engine mock --turns 20 --out bench-results/

# a cascade from registry specs (any registered provider works)
van bench latency --stt mock --llm mock --tts mock --vad energy

# provider options inline (YAML/JSON flow mapping) ...
van bench latency --engine '{provider: mock, response_delay: 0.3}'

# ... or from the same config file `van run` uses (flags override it)
van bench latency --config agent.yaml --scenario benchmarks/scenarios/latency-conversational.yaml \
    --turns 40 --sessions 3

van bench report bench-results/<run-id>   # re-render report.md from the result files

# T2: any STT on pinned LibriSpeech / FLEURS smoke subsets (downloaded once, ~80 MB)
van bench asr --stt faster-whisper/base
van bench asr --stt sherpa-onnx/nemo-fastconformer-en-80ms --mode streaming

# T4: VADs on a labelled corpus (LibriSpeech smoke + noise), turn detectors on eot-bench,
# and the turn-taking battery on any engine
van bench vad --vad energy --vad silero --vad sherpa-onnx/ten-vad
van bench turns --detector smart_turn
van bench turn-taking -c agent.yaml -s benchmarks/scenarios/turn-taking-local.yaml

# T7: what the runtime itself adds (mocks with known delays, offline, ~2 min)
van bench overhead
van bench overhead --baseline benchmarks/baselines/overhead-ci.json   # the CI gate
```

`van bench latency --help` lists every option (`--warmup-turns`, `--dead-air`,
`--reply-timeout`, `--reference-vad`, `--no-audio`, `--json`, ...).

## How the harness works

```
 scenario.yaml ──► stimuli (pre-rendered once: synthetic / TTS / WAV, -20 dBFS,
                   speech on/offsets annotated on the clean audio)
                          │ 20 ms chunks, real time, one absolute schedule
                          ▼
   CallerEmulator ──► LoopbackTransport(realtime_playout=True) ──► AgentSession ──► engine
        ▲                         │ agent audio played at real-time speed
        └── records user (left) + agent (right) on one perf_counter clock
                          │
                          ▼
   stereo recording ──► reference VAD ──► agent onsets ──► items / summary / report
```

* **Caller.** The user stream is continuous, like a microphone: utterances and the silence
  between them are pushed in 20 ms chunks. Chunk *i* covers `[t0 + 20i, t0 + 20(i+1)] ms`;
  it is delivered when that interval has elapsed and time-stamped with its capture start —
  exactly how a capture device behaves — so engine-reported speech boundaries and the
  harness annotations share one clock. After each utterance the caller keeps streaming
  silence until the agent has answered and gone quiet for `gap_after_reply` (and the
  session is idle), or until `reply_timeout` passes without any agent audio (a *missed*
  turn). It also waits for a greeting to finish before speaking.
* **Recording.** Agent audio is placed at the time it started *playing* (from the
  transport's playout log) and cut at playback clears (barge-in). Saved as
  `artifacts/session-NNN/stereo.wav` with an Audacity label track (`labels.txt`: user
  speech spans, agent onsets with their latency, agent speech) and the session event
  timeline (`timeline.jsonl`).
* **Measurement.** Every user-perceived number comes from the recording; the session's own
  metrics are reported next to it to explain where the time went.

## T1 metrics

Times are milliseconds on the harness clock; a *turn* is one scripted user utterance.

| Metric | Definition |
| --- | --- |
| `v2v_ms` | `t_aon − t_uoff` for every turn that expects a reply (headline) |
| `t_uoff` | annotated end of user speech in the stimulus, placed on the recording clock |
| `t_aon` | agent onset: first 10 ms frame that begins ≥ 100 ms of speech on the agent channel according to the **reference VAD** (clicks and comfort noise do not count), refined to the first sample of that frame reaching the level threshold |
| `first_turn_v2v_ms` | `v2v_ms` of turn 0 of every session (cold start); with `--warmup-turns 1` (default) it is excluded from the headline |
| `session_v2v_ms` | the session's own `TurnMetrics.voice_to_voice` (engine end of speech → first agent audio handed to the transport) |
| `residual_ms` | `v2v_ms − session_v2v_ms`: transport/playout buffering, end-of-speech estimation error, leading silence — ≈ 0 on the loopback |
| `eou_delay_ms` | end-of-turn delay (`TurnMetrics.end_of_turn_delay`) |
| `response_ttfb_ms` | turn commit → first agent audio (`TurnMetrics.response_ttfb`) |
| `stt_latency_ms`, `llm_ttft_ms`, `tts_ttfb_ms`, `engine_ttfb_ms` | first component metric of the turn, when the engine reports it |
| `agent_speech_ms` | agent speech (reference VAD) between this turn and the next |
| `session_ready_ms`, `greeting_ms` | `session.start()` duration; greeting onset after ready |
| `dead_air_rate` | share of turns with `v2v_ms` > 2,000 ms (`--dead-air`) **or** no reply |
| `missed_rate` | share of turns without agent speech within `reply_timeout` |
| `premature_rate` | share of turns where the agent started before the user finished (`v2v_ms < 0`) |
| `interrupted_rate` | share of turns whose reply was interrupted |

Distributions report n, mean, std, min, p50/p90/p95/p99 (linear interpolation) and max,
with 95 % percentile-bootstrap confidence intervals (2,000 resamples, fixed seed).

**Reference VAD.** It slightly biases every latency, so it is explicit and configurable:
`--reference-vad rms` (default: 10 ms frames at ≥ −40 dBFS, numpy-only, deterministic,
independent of the engine under test), `rms:-45` (another threshold) or any registered
VAD provider spec (`energy`, `silero` with the `silero` extra). The choice is recorded in
the manifest. Speech models such as Silero do not treat the mock engine's synthetic tone as
speech: keep `rms` for mock engines (the report notes "missed" turns that did play audio).

## Results

`<out>/<run-id>/` (research note 06, §8.5):

| File | Content |
| --- | --- |
| `manifest.json` | suite/schema version, run id, UTC start time, system under test (config as given, credentials redacted, and resolved engine/components/capabilities), scenario definition + hash + SHA-256 of every stimulus, transport, options, onset detector, environment (package versions, git SHA + dirty flag, `uv.lock` hash, CPU model/threads/governor, RAM, GPUs, OS, Python) |
| `items.jsonl` | one line per turn × session with every metric above, flags, transcripts and errors |
| `summary.json` | `metrics` (distributions + CIs), `rates`, `counts`, `extra` (dead-air threshold, `spans_p50_ms`, per-session info incl. caller delivery lag, engine init/warm-up time) |
| `report.md` | Markdown summary, method and per-turn table |
| `artifacts/session-NNN/` | `stereo.wav` (left user, right agent), `labels.txt`, `timeline.jsonl` |

Everything is versioned (`schema_version`); `voice_agent_next.bench.load_run()` reads it
back. `bench-results/` is git-ignored — never commit result dumps.

## Scenarios

Scenarios live in [`scenarios/`](scenarios):

| Scenario | Use |
| --- | --- |
| `latency-smoke` (built-in default) | 10 short questions as synthetic speech, cycled; smoke tier / CI |
| `latency-conversational` | 1–3.5 s turns; endpointing- and STT-sensitive engines |
| `latency-tts` | stimuli synthesized with a TTS provider (swap `mock` for a real voice) |
| `turn-taking-smoke` (built-in) | T4 battery: questions, mid-turn pauses, backchannel, cough, interruption (synthetic) |
| `turn-taking-local` | the T4 battery voiced by Kokoro |

```yaml
name: my-scenario
version: 1
sample_rate: 16000       # user stream rate (transport input format)
chunk: 0.02              # real-time chunk (s)
loudness_dbfs: -20       # speech RMS of every stimulus (null: keep the source level)
lead_in: 0.5             # silence before the first utterance (s)
stimuli: synthetic       # default source: synthetic | tts | wav
tts: kokoro              # TTS spec for `tts` stimuli
reply_timeout: 8.0       # no agent speech this long after the user stopped -> missed
gap_after_reply: 0.3     # speak again once the agent is idle and quiet this long
turns:
  - {id: q1, text: "What time is it?", duration: 0.8}       # synthetic
  - {id: q2, text: "Book a table for two.", source: tts}     # synthesized once
  - {id: q3, wav: data/hello.wav, speech: [0.12, 0.98]}      # WAV + annotated span
  - {id: q4, text: "Mm-hmm.", duration: 0.4, expect_reply: false, pause: 1.0}
  # turn-taking (T4): a turn with a mid-turn pause, and turns spoken over the agent
  - id: q5
    parts: [{text: "Where is my order?", pause: 0.8}, {text: "I placed it last week."}]
  - {id: q6, text: "Uh-huh.", barge_in: 1.5, expect_reply: false, category: backchannel}
  - {id: q7, source: noise, duration: 0.35, seed: 7, loudness_dbfs: -24, barge_in: 1.5,
     expect_reply: false, category: noise}                   # a cough-like burst
```

`parts` renders every part on its own (synthetic or TTS), trims it to its speech and joins
the parts with exactly `pause` seconds of silence; the pauses are annotated. `barge_in: s`
makes the caller speak the turn `s` seconds after the agent's reply to the previous turn
started, over the agent (the reply to such a turn is the agent audio that starts after the
user stopped). `category` labels turns for reports (inferred when omitted: `pause`,
`interruption`, `backchannel`, `noise`, `question`) and `loudness_dbfs` overrides the
scenario level for one turn.

Stimuli are rendered once before a run (fixed resampler, loudness-normalized, padded to
whole chunks) and hashed into the manifest. Without an explicit `speech: [start, end]`,
TTS/WAV clips are annotated automatically (10 ms frames within 35 dB of the loudest
frame). Quote text containing `?`, `:` or `,` inside `{...}` flow mappings.

## Reproducibility and fair comparisons

* **Smoke numbers are regression canaries, not capability scores.** Compare them only with
  a baseline from the same kind of machine.
* For publishable latency numbers use ≥ 100 turns over ≥ 3 sessions per condition
  (`--turns 40 --sessions 3`), keep the default warm-up exclusion, and run one engine at a
  time. Sessions share one engine instance (`engine.warmup()` runs first), like a server.
* The loopback transport includes **no** network, codec, jitter-buffer or device latency;
  cloud engines include the network path to the provider from this machine. State where
  the client, agent and provider run when you publish numbers.
* Check `summary.json → extra.sessions[].push_lag_max_ms`: the caller notes in the report
  when audio was delivered > 50 ms late (event-loop stalls inflate latencies).

## T2: ASR (`van bench asr`)

Accuracy, throughput and latency of any registered STT provider, on the same pinned data:

```bash
van bench asr --stt faster-whisper/base                          # batch, LibriSpeech smoke
van bench asr --stt sherpa-onnx/nemo-fastconformer-en-80ms --mode streaming   # real time
van bench asr --stt '{provider: faster-whisper, model: base, device: cpu}' \
    -d fleurs-en-smoke -d fleurs-es-smoke -d fleurs-de-smoke -d fleurs-tr-smoke -d fleurs-zh-smoke
van bench asr --stt deepgram/nova-3 --mode streaming -d my-calls/manifest.jsonl --language en
van bench asr --stt mock --markdown        # print the per-dataset table (for PRs)
```

**Modes.** `batch` calls `STT.transcribe()` on each whole utterance (what a VAD-segmented
cascade does with a non-streaming recognizer). `streaming` opens `STT.stream()` and pushes
the audio in `--chunk-ms` chunks (default 20 ms), each delivered when its interval has
elapsed like a capture device, at `--realtime-factor` × real time (1 = real time, the
default; 2 = twice as fast; 0 = as fast as possible — then latencies are not real-time
numbers). At the end of the file the harness calls `end_input()` (= `flush()` + end of
input) — exactly what the cascade does when its VAD / turn detector ends the user's turn —
and waits for the final transcript. A batch-only recognizer can be streamed through
`StreamAdapter` with `--vad energy` (or another VAD spec). `--warmup` (default) calls
`stt.warmup()` and transcribes one utterance unmeasured first.

### T2 metrics

| Metric | Definition |
| --- | --- |
| `wer` | **corpus** word error rate Σ(S+D+I) / ΣN over a dataset, after normalization (per dataset in `extra.datasets`; `rates.wer` pools all datasets) |
| `cer` | corpus character error rate; the **headline** for languages written without spaces (zh, ja, ko, th, lo, my, km...), where whitespace is ignored |
| `perfect_rate` | share of utterances without a single error |
| `wer_pct`, `cer_pct` | per-utterance distributions (%) |
| `rtfx` | Σ audio / Σ processing time (batch: `transcribe()` duration; streaming: first chunk → final). Only meaningful in batch mode or with `--realtime-factor 0` |
| `ttfs_ms` | **final latency**: end of audio → final transcript. Streaming: the `end_input()`/`flush()` call → the last `FINAL_TRANSCRIPT` (Pipecat's TTFS, with the end of the file as the VAD stop); batch: the `transcribe()` duration. The STT's share of a voice agent's response time |
| `first_partial_ms` | streaming: capture start of the first chunk → first non-empty interim transcript |
| `interim_revision_rate` | streaming: share of interim updates that rewrite already-shown words instead of appending (the last word may still grow). 0 = interims only grow |
| `processing_ms`, `rtf` | per-utterance processing time and processing / audio |

**Normalization.** Reference and hypothesis go through the same normalizer before
scoring (`--normalizer`, recorded in the manifest):

* `auto` (default): Whisper's `EnglishTextNormalizer` for English — lower case, fillers
  (`uh`, `um`, `hmm`) and bracketed spans removed, contractions and titles expanded
  (`won't` → `will not`, `Mr` → `mister`), spelled-out numbers and currencies to digits
  (`twenty one` → `21`, `$20 million` → `$20000000`), British → American spelling,
  punctuation and diacritics removed. This is the Open ASR Leaderboard convention.
  Other languages: Whisper's `BasicTextNormalizer` keeping combining marks
  (`preserve_marks`, so Indic/Thai vowel signs survive): lower case, punctuation and
  symbols removed. Numbers are **not** normalized outside English.
* `whisper-english`, `whisper-basic` or `none` force one normalizer.

The Whisper normalizers are vendored (`bench/_whisper_normalizer.py`, MIT, identical
output to openai-whisper 20250625), so scoring needs no extra dependency.

### Datasets

| Dataset | Content | Download |
| --- | --- | --- |
| `librispeech-test-clean-smoke` (default) | 50 utterances (1.5–20 s, 409 s) of LibriSpeech test-clean, 10 speakers | first ~80 MB of the 347 MB archive |
| `fleurs-{en,es,de,tr,zh}-smoke` | 10 distinct sentences (2–20 s) per language from the FLEURS test split | 4–8 MB each |
| any `.jsonl` / `.json` / `.tsv` / `.csv` | your own data: `audio` (or `audio_filepath`, `path`, `file`, `wav`) + `text` (or `transcript`, `sentence`), optional `id`, `language`; NeMo manifests work as they are | – |

The built-in subsets are pinned in `src/voice_agent_next/bench/data/asr_smoke.json`:
source archive URL (FLEURS: a fixed Hugging Face commit), and per utterance the archive
member, **SHA-256**, duration and reference text. The source archives are streamed and only
the listed members are kept, verified and cached in `<cache>/datasets/<name>/`
(`$VAN_CACHE_DIR`); the download stops after the last listed member. Later runs are
offline (`VAN_OFFLINE=1` works). The manifest records a dataset hash over ids, references,
languages and audio hashes, plus every file's SHA-256. `--limit N` keeps the first N
utterances. Both corpora are CC BY 4.0 (LibriSpeech: Panayotov et al., 2015; FLEURS:
Conneau et al., 2022). Non-English FLEURS sentences with digits (and Chinese sentences
with Latin letters) are excluded, since only English numbers are normalized.
`benchmarks/tools/make_asr_smoke_subsets.py` regenerates the subsets. FLAC audio
(LibriSpeech) needs `soundfile`: `pip install 'voice-agent-next[bench]'`.

**Reading the numbers.** 50 utterances give a WER confidence interval of roughly ±1–2
points; the FLEURS subsets (10 utterances) are smoke tests for multilingual support, not
rankings. Real-time streaming runs take as long as the audio (~7 min for LibriSpeech).
The Markdown table (`--markdown`, also in `report.md` under "Results by dataset") has one
row per dataset; the report also lists the utterances with the most errors (normalized
reference vs. hypothesis).

## T4: VAD, end-of-turn and turn-taking

Latency alone is misleading: a system that answers in the pause inside "Where is my
order? · I placed it last week." is fast *and* rude. T4 measures turn-taking quality next
to latency, at three levels.

### VAD (`van bench vad`)

```bash
van bench vad --vad energy --vad silero --vad sherpa-onnx/ten-vad       # 50 utterances x 6 conditions
van bench vad --vad silero --condition clean --condition pink@0 --limit 20
```

Every VAD runs over the same **deterministic, frame-labelled corpus**, built from the pinned
LibriSpeech smoke subset (`-d` takes any ASR subset or manifest): utterances normalized to
-20 dBFS, separated by seeded 0.8–2.5 s gaps, 2 s of noise before and 20 s after; one clip
per **condition** with the same layout: `clean` (white noise at -70 dBFS), `pink@20/10/5`,
`white@10` (noise RMS that many dB below the speech), `transient` (keyboard-like clicks up
to -12 dBFS over a -60 dBFS floor). Labels come from the clean utterances: 10 ms frames
within 40 dB of the loudest frame and 10 dB above the noise floor, dips < 150 ms bridged.
The corpus SHA-256 (audio + labels + parameters) is in the manifest and the dataset id.
Audio is streamed through `VAD.stream()` in 20 ms chunks, faster than real time.

| Metric | Definition |
| --- | --- |
| `precision`, `recall`, `f1`, `false_alarm_rate`, `miss_rate` | 10 ms frames; a frame is speech when the (smoothed) probability of the VAD window containing its midpoint reaches the VAD's `activation_threshold` |
| `roc_auc` | over the frame probabilities |
| `onset_lag_ms` | `START_OF_SPEECH` fired − labelled start of the utterance (includes `min_speech_duration`) |
| `offset_lag_ms` | the utterance's last `END_OF_SPEECH` fired − labelled end (includes `min_silence_duration`, the wait before a cascade considers the turn over); negative when the VAD drops the tail |
| `missed_utterances` | utterances without any detected speech |
| `false_alarms_per_min` | `START_OF_SPEECH` outside every utterance (± 100 ms), per minute of the remaining audio |
| `rtf` | inference time / audio time |

VADs that expose only a yes/no decision (the sherpa-onnx Silero and TEN VAD bindings return
`is_speech()`, which already includes sherpa's own hangover) get probabilities of 0 or 1:
their AUC is that of a single operating point, and their frame false alarms include the
hangover after each utterance. Compare them with each other and on F1 / onset / offset.

### End of turn (`van bench turns`)

```bash
van bench turns --detector smart_turn                                   # eot-bench English
van bench turns --detector '{provider: smart_turn, model: v3.2-gpu}' -d eot-bench-es
van bench turns --detector mock -d my-turns.jsonl
```

Follows LiveKit's [eot-bench](https://github.com/livekit/eot-bench) (Apache-2.0). Data:
[`livekit/eot-bench-data`](https://huggingface.co/datasets/livekit/eot-bench-data)
(CC BY 4.0) — real human-to-agent turns, ≤ 400 per language, 14 languages
(`eot-bench-{en,ar,de,es,fr,hi,id,it,ja,ko,nl,pt,tr,zh}`). Each language's Parquet file is
pinned (revision `ca9d98a9`, SHA-256 and size in `bench/eot_datasets.py`; 96–166 MB,
English 162 MB), downloaded once, verified and unpacked into WAV files + `turns.jsonl`
in the cache (unpacking needs `pyarrow`, in the `bench` extra; the Parquet file is then
deleted). Custom data: a JSON Lines manifest with `audio`, `silence_spans` and optional
`words`, `messages`, `language`.

Every silence span ≥ 100 ms of a turn is a decision point: the last one is the end of the
turn (`eot`), the others are mid-turn pauses (`hold`). The detector is asked once per span,
`--score-point` (0.2 s) into the silence, with the causal inputs it would have live: the
turn's audio up to that moment, the previous messages and the words that ended at least
`--transcript-lag` (0.5 s) earlier. Then:

| Metric | Definition |
| --- | --- |
| `false_cutoff_at_300ms`, `false_cutoff_at_600ms` | the lowest false-cutoff rate (hold spans ended by the policy) of any endpointing policy whose mean end-of-turn latency is ≤ 300 / 600 ms |
| `latency_at_5pct_ms`, `latency_at_10pct_ms` (`extra.detector`) | the lowest mean latency (dead air after the user finished) of any policy with ≤ 5 / 10 % false cutoffs |
| policies | `threshold` 0–1 (0.01 steps: fire when `p > threshold`), `action_delay` 0.2–1.0 s (the model fires at `max(action_delay, score time)`), `timeout` 1.0–3.5 s (end the turn anyway); hold spans of 0.2–5 s and all eot spans count |
| `extra.vad_baseline` | the same four numbers for silence alone (answer after a fixed pause) |
| `configured_cutoff_rate`, `extra.configured_policy` | what the cascade does by default: `p ≥ threshold` → answer after `min_endpointing_delay` (0.4 s), else after `max_endpointing_delay` (2.5 s) |
| `accuracy`, `precision`, `recall`, `f1`, `false_positive_rate`, `roc_auc` | complete (eot, positive) vs incomplete (hold) at the detector's threshold |
| `inference_ms` | decision latency: the detector's time per prediction |

The report also lists the Pareto front (`extra.pareto_front`: cutoff rate, latency,
threshold, action delay, timeout).

### Turn-taking battery (`van bench turn-taking`)

```bash
van bench turn-taking                                  # mock engine, built-in smoke scenario
van bench turn-taking -c agent.yaml -s benchmarks/scenarios/turn-taking-local.yaml
van bench turn-taking --engine '{provider: moshi/moshika-q8, url: "ws://localhost:8998"}' \
    -s benchmarks/scenarios/turn-taking-local.yaml
```

The full system — any engine, cascade, native or full-duplex — on the T1 harness (real-time
caller, loopback, stereo recording, reference VAD), with a scenario that mixes plain
questions, turns with **mid-turn pauses** (`parts`) and turns spoken **over the agent**
(`barge_in`): a backchannel, a cough-like noise burst and a real interruption. Everything is
read off the recording; the session's `interrupted` event tells a cut reply from a paused
one.

| Metric | Definition |
| --- | --- |
| `premature_rate` | pause turns in which the agent started before the user finished (agent onset before the end of the last part) |
| `premature_rate_questions` | the same on plain questions |
| `missed_rate`, `dead_air_rate`, `v2v_ms` | as in T1 (`v2v_ms` over questions and pause turns that were not premature) |
| `barge_in_stop_ms` | interruptions: start of the first ≥ 300 ms silence on the agent channel − the user's onset; `stop_within_500ms_rate` |
| `interrupted_rate` | interruptions after which the session cut the reply for good |
| `post_interrupt_response_ms` | next agent onset − end of the interruption |
| `talk_over_ms` | agent speech while the user interrupts |
| `backchannel_yield_rate`, `noise_yield_rate` | the agent went silent ≥ 300 ms within 1 s of the end of the backchannel / noise (a pause-and-resume policy yields briefly by design) |
| `false_barge_in_rate` (`_backchannel`, `_noise`) | the reply was abandoned: the session interrupted it, or the agent stopped and did not speak again before the next user turn |
| `resume_rate`, `resume_gap_ms` | of the yields, those after which the same reply continued, and the silence before it did |

A barge-in turn is scored only if the agent was speaking when it started (`overlapped`);
the report notes the others (make replies longer). The default system is the mock engine
with ~4 s replies; the mock engine gets user transcripts only when it commits a turn, so it
answers backchannels (100 % false barge-ins) — a real STT with the session's backchannel
filter does better.

## T7: framework overhead (`van bench overhead`)

The library claims its runtime adds (almost) nothing to the latency of its components.
T7 measures that claim with mock components whose delays are known, so whatever is left
is the framework. Four sections run in one process, one after the other:

| Section | What runs | Headline |
| --- | --- | --- |
| `micro` | hot paths in a tight loop: resampling (soxr and the numpy fallback), `AudioFrame` helpers, energy VAD, `SilenceTrimmer`, G.711, sentence segmentation, the pre-TTS text filter, `EventEmitter.emit`, `Chan` | µs per operation |
| `e2e` | the T1 harness (real-time caller, loopback, recording, reference VAD) on four mock systems | `overhead_ms`, `frame_jitter_ms`, event-loop lag, CPU |
| `flush` | the app calls `session.interrupt()` 200 ms into every (long) reply | `flush_ms`, leaked frames |
| `capacity` | N concurrent sessions on one engine in one process, N = 1, 2, 4… | `sessions_per_core` |

**Conditions** (`e2e`; `flush` uses `engine` and `cascade`, `capacity` uses `engine`).
`injected_ms` is the latency the configuration prescribes on the critical path from the
end of user speech to the first agent audio:

| Condition | System | `injected_ms` |
| --- | --- | --- |
| `engine` | `MockEngine` (native S2S; energy VAD, 0.4 s silence) | VAD silence = **400** |
| `engine-delay` | `MockEngine`, `response_delay: 0.25` | 400 + 250 = **650** |
| `cascade` | mock STT + LLM + TTS, energy VAD (0.25 s silence), VAD-only endpointing | `max(600, 260 + 0)` = **600** |
| `cascade-delay` | STT latency 0.1 s, LLM TTFT 0.15 s, TTS TTFB 0.1 s, `min_endpointing_delay: 0.2` | `max(200, 260 + 100)` + 150 + 100 = **610** |

The VAD confirms the end of speech in whole windows, so 0.25 s of 20 ms windows is 0.26 s.
The budget (`injected_budget()`) reads these values from the engine the run builds. The
built-in scenario (`overhead-smoke`) uses utterances whose durations are whole 20 ms
chunks, so speech always ends on a VAD window boundary.

### T7 metrics

| Metric | Definition |
| --- | --- |
| `overhead_ms` | `v2v_ms − injected_ms` per turn; `v2v_ms` is measured on the recording exactly as in T1. Reported per condition and pooled (`e2e.overhead_ms`, the headline) |
| `frame_jitter_ms` | standard deviation of `start[i+1] − end[i]` over consecutive agent frames of one reply, as played by the loopback transport (0 = seamless); `frame_gap_max_ms` is the largest gap |
| `loop_lag_ms` | event-loop lag: a probe sleeps 10 ms and records how late it wakes (callbacks that hold the loop + timer granularity). p99 is reported |
| `delivery_lag_ms` | how late the simulated caller delivered its 20 ms chunks (harness, included in `v2v_ms`) |
| `cpu_pct` | process CPU time / wall time while a session ran (100 % = one core; the simulated caller is included) |
| `flush_ms` | interrupt decision (just before `await session.interrupt()`) → end of the last agent audio played; audio playing when the transport clears playback is cut there. Frames that start after the clear are counted as **leaked** |
| `interrupt_call_ms` | duration of the `session.interrupt()` call (engine cancel included) |
| `sessions_per_core` | the largest N whose step passed: overhead p95 ≤ 50 ms, loop lag p99 ≤ 50 ms and no missed turn. One asyncio process is one core. `extra.capacity` also has the CPU share per session and the memory per session (slope of the peak RSS over the steps) |
| `micro.<name>_us` | median over timed batches (garbage collector off) of µs per operation; `% of real time` = cost / audio covered by one operation |

Memory is the resident set size: current on Linux (`/proc/self/statm`) and Windows
(`GetProcessMemoryInfo`), peak (`getrusage`) on macOS. No extra dependencies.

**What remains in `overhead_ms`.** On Linux it is about 1.5–2 ms for the zero-delay
conditions and up to ~5 ms with injected delays. Most of it is asyncio timer lateness:
`epoll` timeouts are rounded up to whole milliseconds, so every mock `sleep()` and every
chunk the caller delivers ends up to 1 ms late (`delivery_lag_ms`). Session, transport
and engine plumbing take well under a millisecond. On Windows with Python < 3.13, asyncio
timers are ~16 ms coarse, so expect larger lag and jitter numbers there. A timer can also
fire up to one tick early, so a mock delay can come in short and a single turn's
`overhead_ms` can be slightly negative. That is why the gate keeps one baseline per OS.

### Tiers

| Tier | e2e | flush | capacity | micro | Wall time |
| --- | --- | --- | --- | --- | --- |
| `smoke` (default, CI) | 4 conditions × 7 turns (1 warm-up) | 2 × 6 interrupts | N ≤ 32, 3 turns per session | 9 batches | ~2 min |
| `full` | 4 conditions × 3 sessions × 36 turns (105 measured) | 2 × 21 interrupts | N ≤ 512 + 3 bisection steps, 5 turns per session | 25 batches | ~15 min |

`--sections`, `--conditions`, `--turns`, `--sessions` and `--max-sessions` override a
tier. A run directory has the usual files. `items.jsonl` holds one line per `turn`,
`session`, `interrupt`, `capacity_step` and `micro` benchmark (see the `kind` field).

## CI regression gate

The `bench overhead gate` job in `.github/workflows/ci.yml` runs the smoke tier on
`ubuntu-latest` and `windows-latest` for every pull request:

```bash
van bench overhead --tier smoke --baseline benchmarks/baselines/overhead-ci.json \
    --retries 1 --summary "$GITHUB_STEP_SUMMARY"
```

`benchmarks/baselines/overhead-ci.json` holds one **entry per runner OS** (`linux`,
`windows`, `darwin`). An entry has the gated statistic of each metric, its 95 % CI, and
the run, commit and machine it came from. A run is judged only against the entry of its
own OS. Without an entry, the job reports and does not gate.

| Rule | Metrics (statistic) | A metric fails when it got worse by |
| --- | --- | --- |
| `latency` | `e2e.overhead_ms`, `e2e.<condition>.overhead_ms`, `e2e.<condition>.v2v_ms`, `e2e.frame_jitter_ms`, `flush.flush_ms` (p50); `e2e.loop_lag_ms` (p99) | more than **10 %** and more than **30 ms**, with non-overlapping 95 % CIs |
| `micro` | `micro.*_us` (median) | more than **200 %** (3×) and more than **5 µs**: runner hardware varies |

Everything else (CPU, memory, capacity, spans) is reported but not gated. A gated metric
that a section should have produced but did not (e.g. every turn missed) fails as
`missing`. Medians over many turns and batches keep a slow turn on a shared runner from
failing the job. With `--retries 1`, a failure also re-runs the failing section: a
metric fails only if it regresses again. Rules live in the baseline file and can be
tuned there.

**Reading the result.** The job summary shows the verdict and a table (baseline, this
run, change, threshold, status: `regressed`, `missing`, `improved`, `new` or `ok`),
followed by the run's tables. The full run (`manifest.json`, `summary.json`,
`items.jsonl`, `report.md`, `gate.json`, `gate.md`) is uploaded as the
`bench-overhead-<OS>` artifact. `improved` means the metric got much better: refresh the
baseline so later regressions are measured from there.

**Updating the baseline.** Only when a change is intended (faster runtime, different
mocks or scenario), or when the runner image changes. Record it on the kind of machine the
entry is for, preferably straight from the CI artifact of the pull request:

```bash
gh run download <run-id> -n bench-overhead-Linux -D ci-linux
van bench overhead --from-run ci-linux/overhead-Linux --update-baseline   # entry `linux`
gh run download <run-id> -n bench-overhead-Windows -D ci-windows
van bench overhead --from-run ci-windows/overhead-Windows --update-baseline   # `windows`
git add benchmarks/baselines/overhead-ci.json   # explain why in the PR description
```

`van bench overhead --update-baseline` on its own runs the smoke tier and records the
entry for the machine you are on. `--platform <key>` picks another entry, and `--baseline
<file>` another file. Other entries and the rules are kept.

## Python API

```python
import asyncio
from voice_agent_next.bench import BenchSystem, LatencyOptions, load_scenario, run_latency_benchmark

system = BenchSystem.from_options(engine={"provider": "mock", "response_delay": 0.3})
results = asyncio.run(
    run_latency_benchmark(
        system, load_scenario("latency-smoke"), LatencyOptions(turns=10), out_dir="bench-results"
    )
)
print(results.summary.metrics["v2v_ms"].p50, results.directory)
```

```python
from voice_agent_next.bench import OverheadOptions, compare_to_baseline, load_baseline
from voice_agent_next.bench import run_overhead_benchmark

results = asyncio.run(run_overhead_benchmark(OverheadOptions.for_tier("smoke", sections=("e2e",))))
print(results.summary.metrics["e2e.overhead_ms"].p50)
gate = compare_to_baseline(load_baseline("benchmarks/baselines/overhead-ci.json"), results)
print(gate.passed, gate.to_markdown())
```

```python
from voice_agent_next.bench.asr_datasets import load_asr_dataset
from voice_agent_next.bench.tracks.asr import AsrOptions, run_asr_benchmark

data = load_asr_dataset("librispeech-test-clean-smoke")
results = asyncio.run(
    run_asr_benchmark(
        "faster-whisper/base", data, AsrOptions(mode="batch"), out_dir="bench-results"
    )
)
print(results.summary.rates["wer"], results.summary.extra["rtfx"])
```

Building blocks for other tracks: `CallerEmulator` (caller), `DuplexRecording` (stereo
recording + labels), `OnsetDetector` / `ProviderReferenceVAD` (onsets), `Distribution` /
`RunManifest` / `RunSummary` / `write_run` / `load_run` (results), `render_report`,
`LoopLagProbe` / `cpu_seconds` / `rss_bytes` (`bench.probes`), `run_micro_benchmarks`
(`bench.microbench`) and the gate (`bench.gate`: `compare_to_baseline`,
`update_baseline`).
