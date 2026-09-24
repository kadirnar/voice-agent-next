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
   │
   └─ speech resumes before the commit → cancel; the same turn continues
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

## Tuning

| Goal | Change |
|---|---|
| Snappier replies | a turn detector (`turn_detector="smart_turn"`), then lower `min_endpointing_delay` (0.2–0.3 s) |
| Fewer cut-offs for slow or thoughtful speakers | raise the detector's `threshold` (more pauses wait for `max_endpointing_delay`) or `min_endpointing_delay` |
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

Each user turn produces a `TurnMetrics` with `end_of_turn_delay` (end of speech → commit),
`response_ttfb` and `voice_to_voice`; the turn detector emits `EOTMetrics` with its
probability and inference time ([observability](observability.md)).
