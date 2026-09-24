# LFM2.5-Audio (`llm: liquid-audio`)

[LFM2.5-Audio-1.5B](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B) by Liquid AI is a
small **omni** model. A 1.2B LFM2.5 language model, a FastConformer audio encoder and a
Mimi-compatible detokenizer let it hear the user's audio and answer with its own speech
(24 kHz) plus a text transcript. The two are generated interleaved, for real time. Its
scores: VoiceBench 54.9 (Moshi: 29.5), and ASR WER 7.5 on average (about Whisper
large-v3). It is English only and uses the LFM Open License v1.0.

In voice-agent-next it is an audio-output LLM: the cascade supplies endpointing (VAD +
turn detector) and the model replaces STT, LLM and TTS. The result is **fully local
native speech-to-speech on a CPU** (see [omni models](../concepts/omni-models.md)).

| | |
|---|---|
| Spec | `liquid-audio/lfm2.5-audio-1.5b` (`llm`) |
| Class | `voice_agent_next.providers.liquid_audio.LiquidAudioLLM` |
| Extra | none: plain HTTP (`httpx`) |
| Server | `llama-liquid-audio-server`, prebuilt from llama.cpp PR [ggml-org/llama.cpp#18641](https://github.com/ggml-org/llama.cpp/pull/18641) (not upstream yet) |
| Runners | Linux x86-64 and arm64, macOS arm64 (and Android). **Windows: none** (use WSL, or a server on another machine) |
| Model | GGUF, pinned revision `7d525f88` of [`LiquidAI/LFM2.5-Audio-1.5B-GGUF`](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-GGUF): Q4_0 ≈ 1.1 GB (default), Q8_0 ≈ 1.8 GB, F16 ≈ 3.3 GB |

## Usage

```python
from voice_agent_next import Agent, AgentSession

session = AgentSession(
    llm={"provider": "liquid-audio", "serve": True},  # downloads + runs the server
    vad="silero",
    turn_detector="smart_turn",
)
```

```yaml
# agent.yaml: no stt, no tts
llm: {provider: liquid-audio, serve: true}
vad: silero
turn_detector: smart_turn
```

With `serve: true` the provider manages the server. On first use it downloads the Q4_0
GGUF set (model, mmproj, vocoder and tokenizer, SHA-256 verified) and the runner for this
platform into the model cache (`$VAN_CACHE_DIR` or the platform cache directory, under
`liquid-audio/`). It then starts `llama-liquid-audio-server` on a free local port during
`warmup()` and stops it on `aclose()`. If the server dies, it is restarted, and the
conversation is replayed, before the next reply.

To run the server yourself (for example on another machine), start it and point the
provider at it:

```bash
python -m voice_agent_next.providers.liquid_audio download          # model + runner
python -m voice_agent_next.providers.liquid_audio serve --port 8080 [--ctx-size 16384]
# or by hand, from the downloaded files:
llama-liquid-audio-server -m LFM2.5-Audio-1.5B-Q4_0.gguf -mm mmproj-LFM2.5-Audio-1.5B-Q4_0.gguf \
    -mv vocoder-LFM2.5-Audio-1.5B-Q4_0.gguf --tts-speaker-file tokenizer-LFM2.5-Audio-1.5B-Q4_0.gguf \
    -c 16384 --port 8080
```

```python
llm = "liquid-audio"  # http://127.0.0.1:8080/v1, or LIQUID_AUDIO_BASE_URL
llm = {"provider": "liquid-audio", "base_url": "http://gpu-box:8080/v1", "context_size": 16384}
```

The server has no authentication: keep it on the loopback interface, or behind your own
proxy.

## Options

| Option | Default | Meaning |
|---|---|---|
| `base_url` | `LIQUID_AUDIO_BASE_URL`, else `http://127.0.0.1:8080/v1` | an external server |
| `serve` | `False` | download and run a managed server |
| `quant` | `"Q4_0"` | `"Q4_0"`, `"Q8_0"` or `"F16"` (with `serve`) |
| `server_options` | `{}` | `LiquidAudioServer` arguments: `threads`, `ctx_size` (default 16384), `port`, `host`, `args`, `executable` (your own build), `startup_timeout` |
| `max_tokens` | 1024 | generation steps per reply (text tokens + 80 ms audio frames) |
| `context_size` | the managed server's `ctx_size`, else 4096 | the server's `-c` (see below) |
| `assistant_note` | `"(Earlier you said: {text})"` | how earlier replies are replayed after a reset; `None` drops them |
| `max_replay_turns` | 8 | most recent messages replayed after a reset |
| `trim_leading_silence` | −45 dBFS | drop the near-silence that opens every reply (see [Latency](#latency)); `None` keeps it |

`LiquidAudioServer` and `download_model()` / `download_runner()` can be used on their own
(`async with LiquidAudioServer() as server: ...`).

## The server's protocol

The server speaks a small subset of OpenAI Chat Completions:

* `POST /v1/chat/completions` with `stream: true` only.
* Roles `system` and `user` only: no `assistant` messages, no tools. User content is text
  or `input_audio` WAV parts; the provider sends the user's turn as 16 kHz WAV.
* The system prompt picks the mode and must be one of a fixed set. The provider always
  sends `Respond with interleaved text and audio.` **The agent's instructions cannot be
  sent** (a warning is logged once).
* The stream carries `delta.content` (text) and `delta.audio_chunk` = `{"data": base64
  float32 PCM, "format": "pcm", "sample_rate": 24000}`. That is one 80 ms chunk per audio
  step, in groups of 12 after every 6 text tokens, so the text runs well ahead of the
  audio. The stream ends with `finish_reason: "stop"` and `data: [DONE]`. The upstream
  PR's int16 `delta.audio` variant is parsed too.
* Sampling is greedy, so `temperature` has no effect.

## The server's stateful context

The server keeps one conversation in its context. Each request **appends** its messages
unless `reset_context: true` is set (the server's default), and the server's own replies,
audio frames included, stay in the context. The provider tracks what the server holds:

* **In sync:** the history is the server's context plus new user turns. Only the new
  turn is sent (`reset_context: false`), so there is no re-prefill (first audio
  ≈ 0.45 s instead of ≈ 0.85 s after a reset).
* **Diverged:** a reply was cut by a barge-in (the history keeps only the heard part), a
  request failed or was cancelled, the history was edited, or a (re)started server knows
  nothing. The provider then resets the context and **replays** the most recent turns.
  User turns are replayed as they were (audio or text); the agent's replies are replayed
  as heard, as user notes (`assistant_note`), because the server rejects assistant
  messages.
* **A cancelled request must be followed by a reset.** After an abort, the server keeps a
  stop flag that makes every non-reset request return nothing, and a leaked audio buffer
  then breaks all later requests (`failed to run prefill`). The provider always resets
  after an incomplete reply. If a non-reset request still fails before any output, it
  retries once from scratch.
* **The server exits when its context is full** (`failed to find a memory slot`). A
  spoken reply costs about 13 positions per second of audio: a 15 s answer takes ~800 of
  them. The provider estimates the context in use and resets and replays before a request
  could overflow `context_size`. A managed server runs with `-c 16384` (the model was
  trained with 128k). An external server's default is 4096, which fills after about four
  long turns: start it with `-c 16384` and pass `context_size`.

## Latency

T1 benchmark (`benchmarks/scenarios/latency-local-omni.yaml`: six questions voiced by
Kokoro, real-time paced, Silero VAD + Smart Turn v3.2). Hardware: AMD Ryzen 5 5600
(6 cores / 12 threads) with the Linux x64 runner, Q4_0, **CPU only**. The prebuilt Linux
runner has no GPU backend, so the RTX 5070 Ti was idle.

MEASUREMENTS

Where the time goes (p50): endpointing ≈ 400 ms (Silero pause + Smart Turn +
`min_endpointing_delay`), then the model. Prefilling the user's audio and generating the
first 80 ms of audio take ≈ 450–500 ms. After that the model opens every reply with
0.4–0.9 s of near-silence (−52…−67 dBFS) before it starts to speak.
`trim_leading_silence` drops that silence: it cannot be skipped entirely, because it has
to be generated first, but it is generated about 2× faster than real time, so trimming it
removes roughly half of it from the voice-to-voice latency.

Generation runs at ≈ 38 steps/s on this CPU, about 2× real time for audio. A reply is
therefore fully generated about halfway through its playback, and barge-in truncation
usually happens after generation has finished.

```bash
van bench latency --scenario benchmarks/scenarios/latency-local-omni.yaml \
    --llm '{provider: liquid-audio, serve: true}' --vad silero --turn smart_turn --turns 24
```

## Limitations

* No tool calling and no custom instructions (fixed system prompt). English only.
* The model's voice is fixed: `Agent(voice=...)` does not apply.
* Its memory across turns is weak (a 1.5B model). Replays after a barge-in help it keep
  the thread, but it does not see its earlier replies as its own.
* Replies are often long (10–15 s). Barge-in works as for any engine.
* The runners come from a work-in-progress llama.cpp PR. The protocol may change when
  it lands upstream.
