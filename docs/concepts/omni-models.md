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
| Qwen3.5-Omni (Alibaba Model Studio) | [`dashscope`](../providers/openai-compatible.md#dashscope-qwen-omni) + `voice=` | cloud | pcm16, 24 kHz |
| Qwen2.5/3-Omni on vLLM-Omni | [`vllm_omni`](../providers/openai-compatible.md#vllm-omni) + `voice=` (experimental) | self-hosted GPU | pcm16 |
| any OpenAI-compatible server with `modalities: ["text", "audio"]` | `openai` + `base_url`, `extra={"modalities": [...]}` | | pcm16 |

Models that hear audio but answer in text (Ultravox, Voxtral, Gemma audio, Qwen2-Audio,
LFM2.5-Audio on `llama-server`...) keep a TTS: that is the
[half-cascade](#audio-input-half-cascade) below.

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

## Audio input: half-cascade

Without `stt=`, the user's turn goes to the LLM as audio even when the LLM answers in
text, and a TTS speaks the reply:

```
user audio ─► VAD ─► turn detector ─► commit ─► audio-input LLM ─► text ─► TTS ─► speaker
                                          └──► (optional) transcript for the history
```

| Host | Provider | Audio sent as |
|---|---|---|
| OpenAI `gpt-audio*`, `gpt-4o-audio*` | `openai` | WAV at the input rate |
| Alibaba Model Studio (Qwen3.5/3.8-Omni) | `dashscope` | WAV 16 kHz as `data:;base64,`, always streamed |
| vLLM (Qwen2-Audio, Qwen-Omni thinker, Ultravox, Voxtral, Gemma 3n...) | `vllm` | WAV 16 kHz |
| vLLM-Omni (Qwen2.5/3-Omni) | `vllm_omni` | WAV 16 kHz; text out unless `voice=` |
| llama.cpp `llama-server` + mtmd (Ultravox, Voxtral, Qwen2.5-Omni, Gemma 4 E2B/E4B, LFM2.5-Audio) | `llamacpp` + `audio_input=True` | WAV 16 kHz |
| Gemini | [`google`](../providers/google.md) | native adapter |

```python
from voice_agent_next import AgentSession, CascadeOptions

session = AgentSession(
    llm={"provider": "llamacpp", "audio_input": True, "audio_history": 2},
    tts="kokoro",
    vad="silero",
    turn_detector="smart_turn",
    cascade_options=CascadeOptions(input_transcriber="llm"),
)
```

* **Does the model hear audio?** `LLMCapabilities.audio_input`. The OpenAI-compatible
  hosts set it from `audio_input=`, or from a table of known audio models matched on the
  model id. A model discovered from the server (`llamacpp`, `vllm` without a model) needs
  `audio_input=True`; `llamacpp` checks the server's `/props` at warmup and warns on a
  mismatch.
* **The user's words for the history.** The session history holds an empty user turn
  unless `CascadeOptions.input_transcriber` provides one: `"llm"` asks the same model in a
  second, text-only request (`llm.transcribe()`; LFM2-Audio gets its own `Perform ASR.`
  prompt, since it answers under any other), and an STT spec such as
  `"faster_whisper/tiny"` transcribes locally. The reply does not wait for it: the
  transcript arrives as the turn's final `user_transcript`, fills the history item and
  `AudioContent.transcript`. With a single-slot local server, start it with
  `-np 2` so the two requests run in parallel.
* **Request size.** Every earlier user turn is audio too. `audio_history=N` on the LLM
  sends only the last N clips as audio and older ones as their transcripts.
* **Preemptive generation** needs an STT transcript to compare, so a half-cascade never
  speculates.

### Measured latency

T1 (`van bench latency`, scenario `latency-local-omni`, 2 sessions × 12 turns, first turn
of each session excluded), 2026-09-25. LFM2.5-Audio-1.5B Q4_0 on mainline `llama-server`
(CUDA build b10567, `-np 2`, RTX 5070 Ti; the model hears the audio and answers in text)
→ Kokoro v1.0 (CPU) · Silero VAD · Smart Turn v3.2. No STT. The machine was shared with
other jobs (load average up to 48 during the first run), which shows in the p90s.

| configuration | v2v p50 | v2v p90 | LLM TTFT p50 | TTS first audio p50 | user transcripts |
|---|---:|---:|---:|---:|---|
| audio only | 1,159 ms | 3,564 ms | 200 ms | 424 ms | none |
| `input_transcriber="llm"`, `audio_history=1` | 1,198 ms | 2,211 ms | 224 ms | 406 ms | 24/24 exact |

The end-of-turn delay (401 ms) and Kokoro on the CPU account for most of the latency.
The model's time to first token includes encoding the audio (≈ 200 ms).
The transcription request runs alongside the reply and does not delay it.

## Limits

* Omni models are large for their quality. LFM2.5-Audio-1.5B is the smallest capable
  one: English only, no tool calling, and a fixed system prompt (the agent's instructions
  are not sent to the model).
* The model's voice is its own: `Agent(voice=...)` and TTS options do not apply
  (`gpt-audio` takes `voice=` on the LLM).
* The text runs ahead of the audio, so the live transcript (`agent_transcript` events)
  leads what the user hears.
