# Alibaba Qwen3-TTS (`qwen-tts`)

[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) is Alibaba's open text-to-speech family:
0.6B and 1.7B models, ten languages (Chinese, English, Japanese, Korean, German, French,
Russian, Portuguese, Spanish, Italian), 24 kHz mono. The models generate 12.5 Hz speech
codes, 16 codebooks per 80 ms frame, which a causal codec turns into audio.
voice-agent-next runs them locally with PyTorch, and **streams**: the first audio of a
sentence plays about 120 ms after it was sent on an RTX 5070 Ti.

| Model | Spec | Voices | Download |
|---|---|---|---:|
| **0.6B CustomVoice** (default) | `qwen-tts`, `qwen-tts/0.6b-custom` | nine built-in speakers, optional style instruction | ~2.5 GB |
| 1.7B CustomVoice | `qwen-tts/1.7b-custom` | same | ~4.3 GB |
| 0.6B Base | `qwen-tts/0.6b` | zero-shot cloning from 3+ s of reference audio | ~2.5 GB |
| 1.7B Base | `qwen-tts/1.7b` | same | ~4.3 GB |
| 1.7B VoiceDesign | `qwen-tts/1.7b-design` | the voice is described in words | ~4.3 GB |

A Hugging Face repository id or a local directory works as the model too
(`qwen-tts/me/my-finetune`).

| | |
|---|---|
| Class | `voice_agent_next.providers.qwen_tts.QwenTTS` (alias `qwen3-tts`) |
| Extra | `voice-agent-next[qwen-tts]` (`qwen-tts>=0.1.1`, `torch>=2.7`, `torchaudio`) |
| Platforms | Linux and Windows (CUDA or CPU), macOS on Apple silicon (MPS or CPU); a GPU is needed for real time |
| Python | 3.11 to 3.14 |
| Output | 24 kHz, mono, s16le, streamed in chunks of 80 ms frames (first chunk 2 frames, then 6) |
| Streaming | audio out: yes (see [How the streaming works](#how-the-streaming-works)). Text in: none, so `stream()` uses the `SentenceStreamAdapter` |
| Word timings | estimated, per sentence (like [Pocket TTS](pocket-tts.md#word-timings)) |
| Device | `device="auto"`: CUDA, then MPS, then CPU (`voice_agent_next.hardware.select_torch_backend`) |
| License | code and weights **Apache-2.0** |

## Setup

```bash
uv sync --extra qwen-tts    # in a checkout: CUDA torch (PyPI on Linux, cu130 index on Windows)
pip install 'voice-agent-next[qwen-tts]'
```

The extra conflicts with `pocket-tts` (CPU torch) and `chatterbox` (`qwen-tts` pins
`transformers==4.57.3`, chatterbox `5.2.0`): install them in separate environments. In this
repository, `[[tool.uv.dependency-metadata]]` drops `qwen-tts`'s `gradio` dependency (its
web demo). On import, qwen-tts prints "SoX could not be found!" when the `sox` program is
missing; it is only used by the 25 Hz tokenizer, which the released 12 Hz models do not use. Install
`flash-attn` and pass `attn_implementation="flash_attention_2"` to use FlashAttention 2.

## Usage

```python
from voice_agent_next import create

tts = create("tts", "qwen-tts")  # 0.6B CustomVoice, speaker "ryan"
tts = create("tts", "qwen-tts/1.7b-custom", voice="vivian", instruct="Speak cheerfully")
tts = create("tts", "qwen-tts/0.6b", voice="me.wav", ref_text="What I say in me.wav.")
tts = create("tts", "qwen-tts/1.7b-design", voice="A calm, low male voice", language="en")
await tts.warmup()  # download + load the model, capture the CUDA graph

async for chunk in tts.synthesize("Chunks arrive while the sentence is generated."):
    play(chunk.frame)
```

In a config file:

```yaml
tts: {provider: qwen-tts/0.6b-custom, voice: aiden, language: en}
```

### Options

| Option | Default | Meaning |
|---|---|---|
| `voice` | `ryan` (CustomVoice) | CustomVoice: `ryan`, `aiden` (English), `vivian`, `serena`, `uncle_fu`, `dylan`, `eric` (Chinese), `ono_anna` (Japanese), `sohee` (Korean). Base: a reference audio file. VoiceDesign: a description |
| `ref_text` | – | Base: transcript of the reference. With it the model continues the reference (in-context, best similarity); without it only the speaker embedding is used |
| `instruct` | – | CustomVoice: style instruction ("Speak slowly and calmly") |
| `language` | auto-detect | `en`, `zh`, `ja`, `ko`, `de`, `fr`, `ru`, `pt`, `es`, `it` or the English name. Setting it avoids mis-detection on short sentences |
| `device` | `auto` | `auto`, `cuda`, `cuda:<n>`, `mps`, `cpu` |
| `dtype` | `auto` | bfloat16 on CUDA, float32 elsewhere |
| `attn_implementation` | `sdpa` | or `flash_attention_2` |
| `first_chunk_frames` | 2 | 80 ms frames in a sentence's first chunk (fewer: sooner first audio) |
| `chunk_frames` | 6 | frames in each following chunk |
| `context_frames` | 25 | frames decoded again as left context with every chunk (seamless joins) |
| `temperature`, `top_k`, `top_p`, `repetition_penalty` | model's generation config (0.9, 50, 1.0, 1.05) | sampling |
| `max_new_tokens` | 60 + 4 per character | frame cap per sentence (a safety net against run-on generation) |
| `cuda_graphs` | `True` | run the per-frame code predictor as a CUDA graph (see below) |

## How the streaming works

`qwen-tts` 0.1.1 renders a whole utterance, then decodes it. This provider:

1. registers a forward hook on the *talker* (the transformer generating one frame per
   step), which hands each new 16-code frame to the provider as it is generated;
2. decodes the frames in chunks with the codec's causal decoder, each chunk with 25 frames
   of left context (for a cloned voice with a transcript, the reference codes are that
   context at the start), exactly like the package's own chunked decoding, so chunks join
   without clicks;
3. skips the package's final whole-utterance decode, which the stream already did;
4. on barge-in, the hook raises at the next step and the generation ends.

**CUDA graphs.** For every frame the talker calls a 5-layer *code predictor* 15 times, once
per residual codebook, through Hugging Face `generate`. In eager PyTorch that costs ~65 ms
per 80 ms frame, almost all of it Python and kernel-launch overhead, so the model ran
slower than real time (RTF 1.4–2.2 on the RTX 5070 Ti). The provider captures that 15-step
loop, sampling included, into one CUDA graph over a static KV cache and replays it
(~10 ms per frame): RTF 0.45. Pass `cuda_graphs=False` to turn it off; other shapes,
greedy decoding or a failed capture fall back to the original `generate` automatically.

## Performance

RTX 5070 Ti (16 GB, Blackwell, driver 615.71), Ryzen 5 5600, torch 2.14 + CUDA 13.0,
Linux. `synthesize()` of three short customer-service sentences (3–6 s of audio each),
after `warmup()`, 3 rounds; first audio p50 (p90) and real-time factor (lower is faster).
Kokoro and Pocket TTS on CPU for comparison. The machine was shared with other workloads
(load average ~7 on 12 threads), which hurts the CPU models most.

| Model | Device | First audio p50 (p90) | RTF |
|---|---|---:|---:|
| Qwen3-TTS 0.6B Base, cloned voice | CUDA | **113 ms** (116) | 0.45 |
| Qwen3-TTS 1.7B CustomVoice | CUDA | **118 ms** (123) | 0.46 |
| Chatterbox Nano | CUDA | 466 ms (658) | 0.15 |
| Chatterbox Turbo | CUDA | 980 ms (1,405) | 0.29 |
| Pocket TTS | CPU | 248 ms (302) | 0.67 |
| Kokoro v1.0 fp16 | CPU | 1,727 ms (2,123) | 0.41 |

The 0.6B and 1.7B models run at the same speed: what remains is the talker's own
`generate` loop, which is launch-bound too. The transcripts of all outputs (faster-whisper
`small.en`) matched the input text word for word.
