# Endpointing: when is the user done?

After the user stops speaking, the cascade waits some silence, the **endpointing delay**
(measured from the end of speech), and then commits the turn and replies. If the user
speaks again before that, the turn goes on. A short delay makes the agent fast. It also
cuts off users who pause mid-thought ("I'd like to book a table for… two people").

```
end of speech ─ VAD pause (0.25 s) ─ final transcript + turn detector ─ ··· delay ··· ─ commit
```

With a streaming or GPU STT, the final transcript arrives 50–100 ms after the end of
speech (`docs/benchmarks/results.md`). The end-of-turn delay then equals the endpointing
delay, so the policy that picks this delay is the next latency lever.

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

## Metrics

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

LOCAL_TABLE

Smart Turn judges "I would like to book a table for" complete on this synthetic voice, so
both policies answer that fragment every time. Neither policy can fix a confident wrong
verdict; only dictation mode or a better detector can. The end-of-turn delay is where
the policies differ. The v2v numbers are dominated by Kokoro int8's first audio
(1.2–1.7 s on the shared CPU) and vary a lot between runs.

## Limitations

* The hold delay only learns from pauses the VAD reports (≥ its `min_silence_duration`),
  and from false commits within `false_commit_window`. A user who is cut off and then
  waits for the agent to finish before correcting it is not learned from.
* A false commit is any speech within the window after a commit. Echo of the agent's
  first words, or a quick "uh-huh", also counts. Keep echo cancellation on.
* Changing the policy or dictation mode affects the next pause, not a pending one.
* The dynamic policy does not look at the transcript itself (trailing "and", digits…).
  That is the turn detector's job.
