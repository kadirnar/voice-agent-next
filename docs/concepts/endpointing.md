# Endpointing

Endpointing is the cascade's decision to **commit** the user's turn: stop listening for
more and start answering. It runs at every candidate pause the VAD reports and is the
largest single contributor to voice-to-voice latency in a cascade, so it is worth tuning.
Native engines endpoint on the server ([turn-taking](turn-taking.md#native-engines)).

## The algorithm

All delays are measured from the **end of speech** (where the VAD saw the voice stop), not
from when the pause was confirmed:

```
speech ends ──► VAD END_OF_SPEECH (min_silence_duration later: a candidate pause)
   │
   ├─ flush the STT: force-finalize, wait ≤ final_transcript_timeout for the final text
   ├─ turn detector: p = P(user is done)        (audio detectors run during the flush)
   │
   ├─ p ≥ threshold (or no detector)  → commit at  min_endpointing_delay
   └─ p <  threshold                  → commit at  max_endpointing_delay
                                        (the fixed policy; see "Endpointing policies")
   │
   ├─ speech resumes before the commit → cancel; the same turn continues
   ├─ speech has begun but the VAD has not confirmed it yet → wait for the VAD's verdict
   └─ the user spoke over the agent → wait for the session's interruption verdict
```

* Without a turn detector the cascade can't tell a finished sentence from a hesitation,
  so it waits longer: `min_endpointing_delay` defaults to **0.6 s** without a detector and
  **0.4 s** with one.
* `max_endpointing_delay` (**2.5 s**) bounds how long a user who is "probably not done"
  can pause before the agent answers anyway.
* An STT `END_OF_TURN` event (Deepgram Flux, AssemblyAI, Cartesia Ink) commits
  immediately and replaces this procedure.
* If the STT doesn't deliver the final transcript within `final_transcript_timeout`
  (1 s), the interim text is used.
* A pause with no transcript at all (noise) commits nothing.
* The VAD confirms speech only after `min_speech_duration` (0.1 s) plus a window, about
  0.15 s after the user resumes. A commit falling in that gap would answer over them, so
  while the speech probability is above the activation threshold without a confirmed
  start, the commit waits (at most `min_speech_duration` + 0.1 s): a confirmed start
  cancels it, silence lets it go through.
* While the user speaks over the agent, the session defers commits until its
  [interruption policy](interruptions.md#engines) has decided: a backchannel is dropped
  instead of being committed and answered.

## Tuning

| Goal | Change |
|---|---|
| Snappier replies | a turn detector (`turn_detector="smart_turn"`), then lower `min_endpointing_delay` (0.2–0.3 s) |
| Fewer cut-offs for slow or thoughtful speakers | `endpointing: dynamic` (learns the user's pauses), raise the detector's `threshold` or `min_endpointing_delay` |
| Numbers, addresses, dictated notes | [dictation mode](#dictation-mode) |
| Users who pause between complete sentences | `min_endpointing_delay` ~1 s with `preemptive_tts` ([measured](#the-local-presets-issue-113)) |
| No 2.5 s silences when the detector misjudges short answers ("Yes.") | `max_endpointing_delay: 1.5` (the local presets) |
| Hide the LLM's time to first token | [preemptive generation](preemptive-generation.md): the reply starts during the endpointing silence and is released at the commit |
| Faster final transcripts | a streaming STT with forced finalization (sherpa-onnx, Moonshine, Deepgram, AssemblyAI) |

```yaml
turn_detector: smart_turn
cascade:
  min_endpointing_delay: 0.3
  max_endpointing_delay: 2.0
  preemptive_generation: true
```

Measure the effect with `van bench latency` ([methodology](../benchmarks/methodology.md)):
`end_of_turn_delay` in the turn metrics is exactly this delay, and `voice_to_voice` is what
the user experiences.

## Metrics

Every endpointing decision emits an `EndpointingMetrics` ([below](#endpointing-metrics)).
Each user turn produces a `TurnMetrics` with `end_of_turn_delay` (end of speech → commit),
`response_ttfb` and `voice_to_voice`; the turn detector emits `EOTMetrics` with its
probability and inference time ([observability](observability.md)).

With a streaming or GPU STT the final transcript arrives 50–100 ms after the end of
speech (`docs/benchmarks/results.md`). The end-of-turn delay then *is* the endpointing
delay, and the policy that picks it is the next latency lever.

## Endpointing policies

`CascadeOptions` has three policies:

| policy | delay |
|---|---|
| `endpointing="fixed"` (default) | `min_endpointing_delay` (0.4 s with a turn detector, 0.6 s VAD-only); `max_endpointing_delay` (2.5 s) when the detector says the user is probably not done |
| `endpointing="dynamic"` | follows the detector's confidence and the user's own mid-turn pauses, within `[min, max]` |
| `dictation=True` | long pauses expected: `dictation_max_delay` (5 s) unless the detector is confident, never less than `dictation_min_delay` (1 s) |

STTs that decide the end of turn themselves (Deepgram Flux, AssemblyAI, Cartesia Ink…:
`capabilities.end_of_turn`) own the commit. None of these policies applies to them.

```python
from voice_agent_next import AgentSession, CascadeOptions

session = AgentSession(
    stt="sherpa-onnx/zipformer-en-kroko",
    llm="ollama/LiquidAI/lfm2.5-1.2b-instruct:latest",
    tts="sherpa-onnx/kokoro-multi-lang-v1_0-int8",
    vad="silero",
    turn_detector="smart_turn",
    cascade_options=CascadeOptions(endpointing="dynamic"),
)
```

or `cascade: {endpointing: dynamic}` in a config file.

## Dynamic endpointing

The delay at a pause depends on the turn detector's probability *p* and its threshold θ
(0.5 for Smart Turn):

```
delay
2.5 s ┤●                                  ceiling: max_endpointing_delay
      │  ●
      │     ●
 hold ┤        ●────────●                 hold: learned from the user's pauses
      │                    ●
0.25 s┤                       ●           floor: min_endpointing_delay
      └┬───────────────┬───────┬─ p
       0               θ       1
```

* **Confident** (*p* ≥ θ): the delay goes linearly from the *hold* delay (at θ) down to the
  floor (at *p* = 1). The floor is 0.25 s with a turn detector: the turn is committed as
  soon as the VAD's candidate pause and the final transcript are in.
* **Likely mid-thought** (*p* < θ): the delay goes from the hold delay up to the ceiling
  (at *p* = 0).
* **No turn detector**: the delay is the hold delay.

The **hold** delay is a pause length the user rarely exceeds mid-turn. Before any pause has
been observed, it is the fixed policy's delay (0.4 s / 0.6 s). After that, it is learned
the way TCP learns its retransmission timeout (RFC 6298): a smoothed mean and a smoothed
mean deviation of the user's mid-turn pauses, and `hold = mean + pause_deviations × dev`,
clamped to `[min, max]`. Two kinds of pause are learned:

* a pause the user resumed from before the commit (the turn went on);
* a **false commit**: the turn was committed, but the user spoke again within
  `false_commit_window` (1 s). The user was probably cut off, so this longer pause raises
  the hold delay.

| option | default | meaning |
|---|---:|---|
| `min_endpointing_delay` | `None` | lower bound; `None` = 0.25 s with a turn detector, 0.3 s VAD-only |
| `max_endpointing_delay` | 2.5 s | upper bound |
| `pause_deviations` | 2.0 | hold = mean + *k* × mean deviation |
| `pause_alpha` | 0.25 | weight of each new pause in the mean and the deviation |
| `false_commit_window` | 1.0 s | speech this soon after a commit makes it a false commit |

The learned pauses belong to the connection (one per session). Changing the policy at
runtime keeps them.

### Cut-off guard

An audio turn detector judges *prosody*: Smart Turn v3.2 gives 0.97 to "Where is my
order?" and 0.99 to "Please change my flight." whether or not the user goes on with "I
placed it last week." Such confident pauses commit at the floor whatever hold delay has
been learned, so the policy above cannot stop cutting off a user who pauses between
sentences.

When the detector said "done" (*p* ≥ θ) and the user went on anyway — a false commit, or a
confident pause they resumed from before the commit — the dynamic policy stops trusting
confident verdicts: confident pauses wait at least the (learned) hold delay. Every commit
the user leaves alone gives trust back: the guard's excess over the floor shrinks by
`pause_alpha` (a quarter) per commit. The fixed policy and dictation mode never use it.

The price is latency after a cut-off. On the T4 battery (below) the guard halves the
premature replies of the local CPU stack (100 % → 50 % of the mid-turn pauses, the first
cut-off of each session is unavoidable) but raises that session's voice-to-voice p50 from
~0.6 s to ~1.5 s: this user pauses a lot, and every question after the first cut-off waits.

## Dictation mode

Phone numbers, card numbers, e-mail addresses and dictated notes contain long pauses
("four five seven … nine two …"). In dictation mode:

* there are no early commits. With a turn detector, the turn ends after
  `dictation_min_delay` (1 s) if *p* ≥ `dictation_threshold` (default: the detector's
  threshold), otherwise after `dictation_max_delay` (5 s);
* without a detector, the turn ends after `dictation_max_delay` of silence;
* pauses in dictation are not learned, because they are not typical pauses.

The defaults (1 s / 5 s) come from the research proposal in
`docs/research/04-turn-taking-vad-interruptions.md`. AssemblyAI's conservative preset uses
0.8 s / 3.6 s. You can turn dictation on for the whole session
(`CascadeOptions(dictation=True)`) or switch it at runtime, typically from a tool:

```python
from voice_agent_next import ToolContext, function_tool


@function_tool
async def start_dictation(ctx: ToolContext) -> str:
    """Call before the user reads out a number, an address or a note."""
    ctx.session.update_endpointing(dictation=True)
    return "Listening; take your time."
```

`session.update_endpointing(mode=..., dictation=...)` (or
`CascadeConnection.update_endpointing`) applies from the next pause on. Engines that do
their own turn detection (native speech-to-speech models) raise `ConfigurationError`.

## Endpointing metrics

Every endpointing decision emits one `EndpointingMetrics` (`session.on("metrics")`):
`policy`, the chosen `delay`, the detector's `probability`/`threshold`, the `hold` delay,
and the outcome. The outcome is either `committed=False` (the user resumed after `pause`
seconds) or a commit, with `false_commit=True` and the `pause` length when the user spoke
again within `false_commit_window`. A commit is reported when its window closes.

`van bench latency` reports these per session (`endpointing_delay_p50_ms`,
`false_commits`, `resumed_pauses`). It also reports a **`cutoff_rate`**: the share of
mid-turn fragments that the agent answered. A fragment is a scenario turn with
`expect_reply: false` followed by a `pause` and the rest of the sentence. See
`benchmarks/scenarios/latency-pauses.yaml` (synthetic) and
`latency-local-sherpa-pauses.yaml` (Kokoro-voiced).

## Choosing the defaults

**Simulation.** We simulated 300 sessions × 40 turns per condition (not committed; the
policy code is the real `Endpointer`):

* mid-turn pauses are lognormal (σ 0.45) around a median of 0.3, 0.45 or 0.7 s;
* pauses shorter than the VAD's 0.25 s are invisible;
* the turn detector's probability is Beta-distributed: mean 0.25 at mid-turn pauses and
  0.85 at turn ends ("good detector"), or 0.35 / 0.7 ("weak detector").

The table shows the end-of-turn delay at real turn ends, and cut-offs per turn, with 30 %
of turns containing a pause:

| detector | user's median pause | fixed: delay p50 / p95, cut-offs | dynamic: delay p50 / p95, cut-offs |
|---|---:|---|---|
| good | 0.3 s | 400 / 400 ms, 0.8 % | **321** / 838 ms, **0.4 %** |
| good | 0.45 s | 400 / 400 ms, 2.1 % | **361** / 1,078 ms, **1.0 %** |
| good | 0.7 s | 400 / 400 ms, 3.3 % | 442 / 1,514 ms, **1.6 %** |
| weak | 0.3 s | 400 / **2,500** ms, 2.1 % | 461 / **1,438** ms, **1.3 %** |
| weak | 0.45 s | 400 / **2,500** ms, 4.2 % | 574 / **1,618** ms, **2.1 %** |
| none (VAD) | 0.3 s | 600 / 600 ms, 1.9 % | 657 / 1,152 ms, 1.6 % |
| none (VAD) | 0.45 s | 600 / 600 ms, 7.7 % | 877 / 1,552 ms, **3.0 %** |
| none (VAD) | 0.7 s | 600 / 600 ms, 19.1 % | 1,298 / 2,146 ms, **4.5 %** |

What the simulation shows:

* **With a good turn detector**, dynamic endpointing commits 40–80 ms sooner at the median
  and halves the cut-offs. A user who pauses long (0.7 s) gets a slightly slower median
  (+40 ms) in exchange for half the cut-offs. The p95 rises because unsure pauses now wait
  a graded 0.8–1.5 s instead of 0.4 s.
* **With a weak detector**, the fixed policy's 2.5 s wait on every missed end dominates
  its tail. Dynamic endpointing grades that wait: p95 drops by ~1 s and cut-offs halve,
  at the cost of a higher median.
* **Without a turn detector**, dynamic endpointing is a safety net, not a speed-up. It
  adapts to users who pause (cut-offs 19 % → 4.5 %) and pays for it in latency. For users
  who hardly pause, the fixed 0.6 s is as good.

The sweep over *k* ∈ {1.5, 2, 3} and α ∈ {0.125, 0.25} picked *k* = 2 and α = 0.25. A
larger *k* cuts off less but waits longer everywhere (*k* = 3 adds ~150 ms). A smaller *k*
brings the VAD-only cut-offs back to fixed levels. α = 0.25 adapts within a few turns and
costs little accuracy compared with 0.125.

We also tried a variant that keeps confident commits between the floor and the fixed
0.4 s, whatever pauses have been learned. It gives a 278 ms median at every pause
length, but cut-offs are 20–50 % higher than the fixed policy. The default favours not
cutting users off.

**`van bench latency` on the mock cascade** (`latency-pauses.yaml`, 48 turns, 16 of them
fragments, back to back; mock STT 100 ms, LLM TTFT 200 ms):

| stack | policy | v2v p50 / p90 | end-of-turn p50 | cut-off rate | false commits |
|---|---|---:|---:|---:|---:|
| energy VAD only | fixed | 606 / 611 ms | 601 ms | 50 % (8/16) | 4 |
| energy VAD only | dynamic | 951 / 1,132 ms | 946 ms | **0 %** (0/16) | 3 |
| + mock turn detector (punctuation) | fixed | 607 / 612 ms | 401 ms | 0 % | 0 |
| + mock turn detector (punctuation) | dynamic | **597** / 645 ms | **391** ms | 0 % | 0 |

VAD-only: with the fixed 0.6 s, the scenario's 0.6–0.8 s pauses are cut every time. The
dynamic policy cut the first one, learned the pause, and waited ~0.9–1.2 s afterwards.
The mock detector is certain about every punctuated transcript, so neither policy cuts
anyone off there, and the learned pauses keep the confident delay near 0.4 s.

**Local CPU stack** (`latency-local-sherpa-pauses.yaml`, 36 turns: 23 measured questions
and 12 fragments; Silero VAD · Smart Turn v3.2 · sherpa-onnx `zipformer-en-kroko`
streaming STT · Ollama `lfm2.5-1.2b-instruct` · sherpa-onnx Kokoro int8; Ryzen 5 5600,
back to back, other jobs running on the machine):

| run | policy | chosen delay p50 | end-of-turn p50 | v2v p50 / p90 | cut-offs (fragments answered) | false commits |
|---|---|---:|---:|---:|---:|---:|
| 1 | fixed | 400 ms | 401 ms | 2,564 / 3,796 ms | 6 / 12 | 0 |
| 1 | dynamic | **323 ms** | **361 ms** | 1,951 / 2,922 ms | 5 / 12 | 0 |
| 2* | fixed | 400 ms | 424 ms | 3,251 / 4,956 ms | 5 / 12 | 0 |
| 2* | dynamic | **324 ms** | 432 ms | 4,251 / 6,348 ms | **3 / 12** | 2 |

\* Run 2 overlapped the full unit-test suite on the same CPU, so its timings (STT,
Smart Turn, TTS) are inflated. Only the chosen delays and the cut-offs are comparable.

Smart Turn judges "I would like to book a table for" complete on this synthetic voice, so
both policies answer that fragment every time. Neither policy can fix a confident wrong
verdict at its first occurrence; only dictation mode or a better detector can (since
#113 the [cut-off guard](#cut-off-guard) keeps the dynamic policy from repeating it). On
the other fragments, dynamic
endpointing answered 2 of 18 fragments over both runs, against 5 of 18 with the fixed
policy. Its chosen delay is ~75 ms shorter at the
median. In run 1 that shortened the end of turn by 40 ms. The v2v numbers are dominated
by Kokoro int8's first audio (1.2–1.7 s, and more on the shared CPU) and vary too much
between runs to show a 40–80 ms change. Use the end-of-turn delay to compare policies.

## The local presets (issue #113)

The T4 battery (`van bench turn-taking -s benchmarks/scenarios/turn-taking-local.yaml`)
found 92 % premature replies in mid-turn pauses, 100 % false barge-ins on "uh-huh" and
29 % dead air on the local CPU cascade. What each number came from:

* **Dead air:** Smart Turn says "not done" at the end of short or clipped answers ("Yes.",
  "For two people tonight."; *p* = 0.02–0.24), and the fixed policy then waited 2.5 s. On
  eot-bench English (400 real turns) it does so at **25.5 %** of the turn ends.
* **Premature replies:** Smart Turn is confident at every mid-turn pause of the battery
  (0.57–0.99): three first parts are complete sentences, and even "I would like to book
  a table," scores 0.88 on this voice (0.46 when the "uh-huh" transcript below had leaked
  into the turn). The reply then starts ~0.55 s into a 0.5–1.0 s pause. Only the 0.5 s
  pause is now safe: the commit waits while the VAD has not yet confirmed resumed speech.
* **Backchannels:** see [interruptions](interruptions.md#short-utterances-small-asr-models).

eot-bench English with Smart Turn v3.2 (int8), the cascade's fixed policy at several
bounds (`van bench turns --detector smart_turn`, re-evaluated per span):

| min / max delay | false cutoffs | mean end-of-turn latency | turn ends waiting > 1.7 s |
|---|---:|---:|---:|
| 0.4 / 2.5 s (default) | 10.8 % | 936 ms | 25.5 % |
| 0.4 / 1.5 s | 13.9 % | 680 ms | 0 % |
| **0.5 / 1.5 s** (local presets) | **10.8 %** | **755 ms** | **0 %** |
| 0.6 / 1.5 s | 8.7 % | 830 ms | 0 % |
| 1.0 / 1.5 s | 6.4 % | 1,128 ms | 0 % |
| dynamic, default bounds 0.25 / 2.5 s (nothing learned yet) | 18.3 % | 683 ms | 16.5 % |

`local-cpu`, `local-gpu` and `apple` therefore use `min_endpointing_delay: 0.5`,
`max_endpointing_delay: 1.5` and preemptive generation with preemptive TTS — the reply is
synthesized during the endpointing silence, so the extra 0.1 s does not reach the user
(`hybrid` gets the bounds without preemptive generation, which would cost cloud tokens
per discarded attempt). See `voice_agent_next.presets.LOCAL_TURN_TAKING`.

**What no default fixes.** A pause after a complete sentence *is* a turn end as far as
the audio can tell. Answering within ~0.6 s of it and never answering inside a 1 s
mid-turn pause are incompatible. On the battery (below), the reply lands in the pause
unless the commit comes after the pause plus the VAD's onset delay:

| local CPU stack, T4 battery | premature | v2v p50 |
|---|---:|---:|
| fixed 0.5 / 1.5 s (local presets) | 75 % | ~0.6–0.7 s |
| dynamic with the cut-off guard | 50 % | ~1.5 s |
| fixed **1.0** / 1.5 s | **0 %** | ~1.1 s |

For users who pause between sentences (dictating, thinking aloud, older users), raise
`min_endpointing_delay` towards 1 s — with preemptive TTS the reply is ready when the
silence ends — or switch to dictation mode while they read something out.

## Semantic end of turn (issue #124)

The remaining premature replies come from pauses after phrases the audio detector is sure
are complete. A text detector fused with it ([`fused`](../providers/lm-turn.md)) moves
the endpointing decision where the words disagree with the sound:

* the fused probability replaces the detector's in every policy (fixed, dynamic,
  dictation); `EndpointingMetrics.audio_probability` / `text_probability` show both
  halves;
* "I would like to book a table," (audio 0.88) reads as unfinished: the fused probability
  falls under the threshold and the fixed policy waits `max_endpointing_delay` (premature
  in 0 of 9 sessions, 5 of 6 with Smart Turn alone);
* the text half runs while the STT flushes and the reply is prepared (a held speculative
  reply starts on the audio verdict, before the text half returns), so a complete turn
  costs little more than with Smart Turn alone.

| local CPU stack (#122), T4 battery | premature | v2v p50 | eot-bench en false cut-offs @ 0.5 / 1.5 s |
|---|---:|---:|---:|
| Smart Turn, fixed 0.5 / 1.5 s (local presets) | 75 % | 644 ms | 10.8 % |
| **fused** (Smart Turn + `lm_turn`), fixed 0.5 / 1.5 s | **50 %** | 626–735 ms | **8.8 %** |

Details, German and Spanish, and T1: [semantic end of turn](../providers/lm-turn.md#measurements).

What it does not fix: "Where is my order?" is a complete question to the ear *and* to
the reader. Only waiting longer after every complete question (a 1.0 s minimum, see
above) avoids answering it before "I placed it last week."

## Limitations

* The hold delay only learns from pauses the VAD reports (≥ its `min_silence_duration`),
  and from false commits within `false_commit_window`. A user who is cut off and then
  waits for the agent to finish before correcting it is not learned from.
* A false commit is any speech within the window after a commit. Echo of the agent's
  first words, or a quick "uh-huh", also counts (and, in the dynamic policy, raises the
  cut-off guard after a confident commit). Keep echo cancellation on.
* Changing the policy or dictation mode affects the next pause, not a pending one.
* The dynamic policy does not look at the transcript itself (trailing "and", digits…).
  That is the turn detector's job: use a text detector
  ([`fused`](../providers/lm-turn.md)) for that.
