# Omni models: audio-output LLMs in the cascade

An *omni* model hears the user's audio and answers with **its own speech**, plus a text
transcript of what it says. It replaces STT, LLM and TTS at once. The cascade still
decides when the user has finished (VAD + turn detector), and the session still does
barge-in, truncation, history and metrics. The result is native speech-to-speech on the
same runtime as everything else:

```python
from voice_agent_next import Agent, AgentSession

session = AgentSession(
    llm={"provider": "liquid-audio", "serve": True},  # LFM2.5-Audio-1.5B, local, CPU
    vad="silero",
    turn_detector="smart_turn",
)  # no stt=..., no tts=...
await session.start(Agent("You are a helpful receptionist."), transport)
```

```
user audio ─► VAD ─► turn detector ─► commit ─► omni LLM ─┬─ audio deltas ─► ResponseAudio ─► speaker
              (the user's turn goes to the model as audio) └─ text deltas  ─► ResponseText (transcript)
```

| Model | Provider | Where | Audio out |
|---|---|---|---|
| LFM2.5-Audio-1.5B (Liquid AI) | [`liquid-audio`](../providers/liquid-audio.md) | local (CPU), `llama-liquid-audio-server` | float32 PCM, 24 kHz |
| `gpt-audio`, `gpt-4o-audio-preview` (OpenAI) | [`openai`](../providers/openai.md) (Chat Completions) | cloud | pcm16, 24 kHz |
| any OpenAI-compatible server with `modalities: ["text", "audio"]` | `openai` + `base_url`, `extra={"modalities": [...]}` | e.g. vLLM-Omni (Qwen-Omni) | pcm16 |

## How it works

* **Capabilities.** An LLM that streams speech declares `LLMCapabilities(audio_output=True,
  audio_sample_rate=24_000)`. Each streamed `ChatChunk` may carry `audio` (an s16le
  `AudioFrame` delta) next to its text `delta`. The model's text is the transcript of
  its speech.
* **Routing.** `CascadeEngine` plays the LLM's audio when the LLM has `audio_output` and
  **no TTS is configured**, or when `CascadeOptions(use_llm_audio=True)` asks for it. The
  audio deltas become `ResponseAudio`, and the text deltas become `ResponseText` without
  a text filter: the model has already said them. With a TTS as well (and
  `use_llm_audio=True`), the TTS still speaks verbatim text such as `session.say()` and
  greetings. Without any TTS, `say()` sends its text as a transcript only, and logs a
  warning.
* **Input.** With `stt=None`, the user's turn reaches the model as `AudioContent` (the
  half-cascade), so the model hears tone and hesitation, not just words. You can also keep
  an STT, for example to get user transcripts, if the model takes text.
* **Metrics.** `TurnMetrics.voice_to_voice` and `response_ttfb` measure the first *audio*,
  as for any engine. `EngineMetrics.ttfb` is the time from commit to the first audio.
  `LLMMetrics.ttft` is the first token of any kind; the new `LLMMetrics.ttfb` is the
  first audio chunk.

## Barge-in and truncation

When the user interrupts, the session tells the engine how much of the reply was played
(`truncate(item_id, audio_end_ms)`). The cascade keeps only the heard part in the history,
marks it `interrupted`, and the model sees that version on the next turn.

* **Reported interleaving (exact).** A model that knows where each text delta is spoken
  sets `ChatChunk.audio_offset` (seconds into the reply's audio). The heard text is then
  every delta whose offset lies before the cut.
* **Proportional estimate (otherwise).** When the reply has been fully generated, the
  heard share of the text is the played share of the audio. The cut is extended to the end
  of the word being spoken. While the model is still generating, the audio so far says
  nothing about the text's length, because omni models write their text *ahead* of their
  audio. LFM2.5-Audio, for example, emits 6 text tokens per 0.96 s of audio, so its text
  is complete when only about 40 % of the audio exists. The reply's length is then
  estimated from the speaking rate: `CascadeOptions.speech_rate`, 14 characters per
  second by default (measured on LFM2.5-Audio), refined from each completed reply.

A stateful server such as `llama-liquid-audio-server` holds the full reply in its
context. The provider notices that the history now differs (the heard part), resets the
server's context and replays the conversation. See
[liquid-audio: context](../providers/liquid-audio.md#the-servers-stateful-context).

## Preemptive generation

For an audio LLM, generating a reply *is* synthesizing speech. The cascade therefore
speculates (`preemptive_generation=True`) only when `preemptive_tts=True` also allows
speculative speech. Otherwise it waits for the commit, as without preemptive generation.
Speculation also needs an STT, since the transcript is what is compared, so a typical
omni setup (`stt=None`) never speculates.

## Limits

* Omni models are large for their quality. LFM2.5-Audio-1.5B is the smallest capable
  one: English only, no tool calling, and a fixed system prompt (the agent's instructions
  are not sent to the model).
* The model's voice is its own: `Agent(voice=...)` and TTS options do not apply
  (`gpt-audio` takes `voice=` on the LLM).
* The text runs ahead of the audio, so the live transcript (`agent_transcript` events)
  leads what the user hears.
