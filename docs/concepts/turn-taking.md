# Turn-taking

A voice agent must decide two things all the time: **has the user finished?** (answer now,
or wait) and **is the user taking the floor?** (stop talking, or keep going). Answering
too early cuts people off mid-thought; answering too late feels sluggish. voice-agent-next
splits the decision into signals of increasing cost and meaning:

| Signal | Component | Cost | What it knows |
|---|---|---|---|
| Voice activity | VAD (`silero`, `sherpa_onnx`, `energy`) | < 1 ms per 32 ms window on CPU | someone is speaking / silence |
| Semantic end of turn | turn detector (`smart_turn`) | tens of ms per pause | the utterance *sounds* finished (audio) or *reads* finished (text) |
| Provider turn events | STT with `end_of_turn` (Deepgram Flux, AssemblyAI, Cartesia Ink, Soniox, Speechmatics) | included in the STT | end of turn, eager end of turn, turn resumed |
| Server turn detection | native engines (OpenAI `semantic_vad` / `server_vad`, Gemini Live) | included in the engine | the engine commits turns itself |

Background and measurements: research note
[04 · Turn-taking, VAD, interruptions](../research/04-turn-taking-vad-interruptions.md).

## VAD: speech and candidate pauses

The VAD turns audio into `START_OF_SPEECH` / `END_OF_SPEECH` events (`VADOptions`):

| Option | Default | Meaning |
|---|---|---|
| `activation_threshold` | 0.5 | Probability at or above which a window counts as speech |
| `deactivation_threshold` | activation − 0.15 | Probability below which a window counts as silence (hysteresis) |
| `min_speech_duration` | 0.1 s | Speech needed before `START_OF_SPEECH` |
| `min_silence_duration` | 0.25 s | Silence needed before `END_OF_SPEECH` |
| `prefix_padding_duration` | 0.5 s | Audio kept before the speech start |

`min_silence_duration` is short on purpose: a VAD end is only a **candidate** pause. The
decision to end the turn belongs to [endpointing](endpointing.md). Speech start is also
what triggers a possible [barge-in](interruptions.md).

## Turn detectors: is the user done?

A `TurnDetector` returns the probability that the user has finished, from the audio of the
current turn (`modality = "audio"`, e.g. [Smart Turn](../providers/smart_turn.md)), the
conversation text (`"text"`), or both. The cascade calls it at each candidate pause; its
`threshold` (0.5 by default) picks the short or the long endpointing delay:

```python
AgentSession(
    stt="deepgram/nova-3", llm="groq", tts="cartesia", vad="silero", turn_detector="smart_turn"
)
```

Audio detectors run in parallel with the STT flush (they don't need the transcript), so
they add little to the turn's latency.

An audio detector judges how the utterance *sounds*: a complete sentence followed by a
pause ("Where is my order? · I placed it last week.") is "done" to it (Smart Turn: 0.97),
and short answers ("Yes.") often sound unfinished. The T4 battery (`van bench
turn-taking`) measures what that does to a whole system; the local presets' settings and
the remaining trade-off are in [endpointing](endpointing.md#the-local-presets-issue-113).

## STT-driven turns

STTs that detect turns themselves set `STTCapabilities.end_of_turn` and emit:

* `END_OF_TURN`: the cascade commits at once, without its own endpointing;
* `EAGER_END_OF_TURN`: the turn has *probably* ended, a
  [preemptive](preemptive-generation.md) reply may start;
* `TURN_RESUMED`: the user kept talking; the speculative reply is discarded.

Leave out `vad=` to let the provider decide both speech and turns, or keep a local VAD for
fast barge-in detection. The provider pages ([Deepgram](../providers/deepgram.md),
[AssemblyAI](../providers/assemblyai.md), [Cartesia](../providers/cartesia.md),
[Soniox](../providers/soniox.md), [Speechmatics](../providers/speechmatics.md)) show both
setups.

## Native engines

Native engines detect turns on the server by default (`server_turn_detection`). The
[OpenAI Realtime](../providers/openai-realtime.md) engine uses `semantic_vad` where the
backend supports it and `server_vad` otherwise; `turn_detection=None` means manual turns
(`commit_input()`). The session behaves the same either way: it reacts to
`InputSpeechStarted`, `InputSpeechStopped` and `InputCommitted`.

## Push-to-talk

`EngineOptions.turn_detection=False` (native engines) or no automatic endpointing: call
`commit_input()` when the user releases the button, and `clear_input()` to drop what was
said.
