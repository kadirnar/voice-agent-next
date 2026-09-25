# The cascade

`CascadeEngine` turns any mix of registered components into a speech-to-speech engine:

```
user audio ─┬─► VAD ──────────► speech start / candidate pause
            └─► STT stream ───► partial / final transcripts (+ STT turn events)
                                   │
                    endpointing: turn detector + silence ──► commit the turn
                                   │
                                  LLM (streaming text, tool calls)
                                   │  sentence segmenter, text filter
                                  TTS stream ──► agent audio, aligned with its text
```

```python
from voice_agent_next import AgentSession, CascadeOptions

session = AgentSession(
    stt="sherpa-onnx/zipformer-en-kroko",
    llm="ollama/qwen3.5:4b",
    tts="kokoro",
    vad="silero",
    turn_detector="smart_turn",
    cascade_options=CascadeOptions(min_endpointing_delay=0.3),
)
```

The same in a config file: `stt:`, `llm:`, `tts:`, `vad:`, `turn_detector:` and a
`cascade:` section with `CascadeOptions` fields.

## A user turn, step by step

1. **Listening.** Audio streams continuously into the STT stream and the VAD. VAD start →
   `InputSpeechStarted` (the session may treat it as a [barge-in](interruptions.md)).
2. **A pause.** VAD end after `min_silence_duration` (0.25 s by default) is only a
   *candidate* end of turn → `InputSpeechStopped`, and [endpointing](endpointing.md)
   starts: flush the STT, wait for the final transcript, score the turn with the turn
   detector, then wait for the rest of the endpointing delay. Speech resuming cancels it.
3. **Commit.** `InputCommitted` and the final `InputTranscript`; the user message joins
   the chat context.
4. **Response.** The LLM streams text. A `SentenceSegmenter` releases complete sentences,
   with a short first chunk for fast first audio (`first_sentence_min_chars`,
   `first_sentence_max_chars`). Each sentence is cleaned (`text_filter`, markdown and emoji
   removal by default) and pushed into the TTS stream. TTS engines that need it rewrite
   numbers, amounts, dates and addresses into words
   ([text normalization](text-normalization.md)); the transcript keeps the LLM's text.
5. **Speech.** TTS audio becomes `ResponseAudio`; the text of each sentence is emitted with
   its audio offset, so an interruption knows exactly which words were heard.

With a non-streaming TTS, the `SentenceStreamAdapter` synthesizes sentence by sentence
with one sentence of prefetch, which gives the same alignment.

## Turn signals from the STT

Some STTs decide the end of turn themselves (Deepgram Flux, AssemblyAI Universal-Streaming,
Cartesia Ink: `capabilities.end_of_turn`). Their `END_OF_TURN` commits immediately and the
cascade's own endpointing is skipped; `EAGER_END_OF_TURN` starts a
[preemptive](preemptive-generation.md) reply and `TURN_RESUMED` discards it. Without a VAD,
the STT's start/end-of-speech events drive the turn instead. See [turn-taking](turn-taking.md).

## Half-cascade

Without an STT, an LLM that accepts audio (`capabilities.audio_input`: Gemini, Qwen-Omni,
gpt-4o-audio, ...) receives the user's turn as `AudioContent`. A VAD is required to
segment turns.

```python
AgentSession(llm="google/gemini-3.8-flash", tts="cartesia", vad="silero")
```

## Options

| `CascadeOptions` field | Default | Meaning |
|---|---|---|
| `min_endpointing_delay` | 0.4 s with a turn detector, 0.6 s without | Silence (from the end of speech) before committing when the user seems done |
| `max_endpointing_delay` | 2.5 s | Silence before committing when the turn detector says the user is probably not done |
| `final_transcript_timeout` | 1.0 s | Max wait for the STT's final transcript after flushing (falls back to the interim text) |
| `text_filter` | `tts_clean` | Applied to each sentence before TTS |
| `first_sentence_min_chars` / `first_sentence_max_chars` | 4 / 40 | Shape the first spoken chunk for fast first audio |
| `max_history_items` | `None` | Truncate the LLM context (the system prompt is always kept) |
| `turn_audio_prefix` | 0.5 s | Audio kept before speech start for the turn detector / audio LLM |
| `preemptive_generation`, `preemptive_tts`, `preemptive_threshold`, `preemptive_max_speech`, `preemptive_max_attempts` | off | [Preemptive generation](preemptive-generation.md) |

Per-component options go to the component (`stt: {provider: deepgram, language: en}`);
the provider pages list them.

## Failover

`stt`, `llm` and `tts` accept a list of specs: a [failover chain](failover.md) that switches
providers only where the user cannot hear it.
