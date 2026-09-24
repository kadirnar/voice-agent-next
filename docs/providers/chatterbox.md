# Resemble AI Chatterbox (`chatterbox`)

[Chatterbox](https://github.com/resemble-ai/chatterbox) is Resemble AI's family of open
text-to-speech models. All of them clone a voice from a few seconds of reference audio and
produce 24 kHz mono speech. voice-agent-next runs them locally with PyTorch, on an NVIDIA
GPU (CUDA), an Apple silicon GPU (MPS) or the CPU. No API key; offline once the weights are
cached.

| Model | Spec | Parameters | Languages | Download | Notes |
|---|---|---:|---|---:|---|
| **Turbo** (default) | `chatterbox`, `chatterbox/turbo` | 350M | English | ~3 GB | built for voice agents: one-step mel decoder, paralinguistic tags (`[laugh]`, `[chuckle]`, `[cough]`...) |
| Nano | `chatterbox/nano` | 110M | English | ~2 GB | Turbo's architecture with a smaller backbone; fast enough for a CPU. Needs chatterbox-tts newer than 0.1.7 (see [Setup](#setup)) |
| Multilingual | `chatterbox/multilingual` | 500M | 23 | ~3 GB | `language="fr"`...; `exaggeration` and `cfg_weight` controls |

| | |
|---|---|
| Class | `voice_agent_next.providers.chatterbox.ChatterboxTTS` |
| Extra | `voice-agent-next[chatterbox]` (`chatterbox-tts>=0.1.7`, `torch`, `torchaudio`) |
| Platforms | Linux and Windows (CUDA or CPU), macOS on Apple silicon (MPS or CPU). Not on Intel Macs (no PyTorch wheels) |
| Python | 3.11 to 3.14 |
| Output | 24 kHz, mono, s16le, one chunk per sentence |
| Streaming | none in the model: long texts are rendered sentence by sentence, `stream()` uses the `SentenceStreamAdapter` |
| Word timings | estimated, per sentence (like [Pocket TTS](pocket-tts.md#word-timings)) |
| Device | `device="auto"`: CUDA, then MPS, then CPU (`voice_agent_next.hardware.select_torch_backend`) |
| License | code and weights **MIT**. Every output carries Resemble AI's imperceptible [Perth](https://github.com/resemble-ai/Perth) watermark, as upstream does; this provider does not remove it |

## Setup

```bash
uv sync --extra chatterbox        # in a checkout: CUDA torch (PyPI on Linux, cu130 index on Windows)
```

The extra conflicts with `pocket-tts` (CPU torch) and `qwen-tts` (another transformers pin):
uv resolves each in its own fork of the lockfile, and `uv sync` refuses to install two of
them together. Use separate environments (`UV_PROJECT_ENVIRONMENT=...`) to compare them.

`chatterbox-tts` 0.1.7 pins `torch==2.6.0`, `numpy<2` (Python < 3.13) and its `gradio` demo.
torch 2.6 has no kernels for Blackwell GPUs (RTX 50xx, compute capability 12.0), and
numpy < 2 conflicts with `kokoro-onnx`. In this repository `[[tool.uv.dependency-metadata]]`
relaxes those pins for uv (torch >= 2.7, any numpy, no gradio); the library runs unchanged
with them (tested with torch 2.14 + CUDA 13.0, numpy 2.5). **pip** reads the published
metadata instead, so on a Blackwell GPU upgrade torch afterwards:

```bash
pip install 'voice-agent-next[chatterbox]'
pip install -U torch torchaudio     # Linux: PyPI's CUDA 13 build; Windows: add
                                    #   --index-url https://download.pytorch.org/whl/cu130
```

`van doctor` shows the torch build and where `device="auto"` puts the model, with the fix
when the installed torch cannot use the GPU (a CPU build, or a build without kernels for it).

The extra also pins `setuptools<81`: Resemble's watermarker (`resemble-perth` 1.0.1) imports
`pkg_resources`, which setuptools 81 removed.

**Chatterbox-Nano** is on the chatterbox `master` branch but not yet in a PyPI release:

```bash
pip install --no-deps -U 'chatterbox-tts @ git+https://github.com/resemble-ai/chatterbox'
```

Without it, `chatterbox/nano` fails with a `ConfigurationError` saying so.

Weights are downloaded on first use into the Hugging Face cache. For Turbo and Nano the
provider downloads only the files it loads: upstream's `from_pretrained` also fetches a
second, unused 1 GB decoder. Turbo and Nano share every file except the T3 backbone.

## Usage

```python
from voice_agent_next import create

tts = create("tts", "chatterbox")  # Turbo, built-in voice
tts = create("tts", "chatterbox", voice="me.wav")  # clone (> 5 s of clean speech)
tts = create("tts", "chatterbox/nano", device="cpu")
tts = create("tts", "chatterbox/multilingual", language="fr", cfg_weight=0.3)
await tts.warmup()  # download + load the model now (~30 s on first run)

audio = await tts.synthesize("Hi there [chuckle], how can I help?").collect()
```

In a config file:

```yaml
tts: {provider: chatterbox/turbo, voice: /path/to/reference.wav}
```

### Options

| Option | Default | Meaning |
|---|---|---|
| `voice` | built-in voice | reference audio file to clone (WAV/MP3/FLAC..., more than 5 s; 10 s works best). Encoded once, then cached (8 voices) |
| `language` | `en` | multilingual model only: `ar da de el en es fi fr he hi it ja ko ms nl no pl pt ru sv sw tr zh` |
| `device` | `auto` | `auto`, `cuda`, `cuda:<n>`, `mps`, `cpu` |
| `model_path` | – | load the weights from this directory instead of Hugging Face |
| `temperature` | 0.8 | sampling temperature |
| `top_p` | 0.95 (Turbo/Nano), 1.0 (multilingual) | nucleus sampling |
| `top_k` | 1000 | Turbo/Nano only |
| `repetition_penalty` | 1.2 (Turbo/Nano), 2.0 (multilingual) | |
| `exaggeration` | 0.5 | multilingual only: emotion intensity |
| `cfg_weight` | 0.5 | multilingual only: classifier-free guidance (lower for fast speakers or a cross-language reference) |
| `norm_loudness` | `True` | Turbo/Nano: normalize a cloned reference to -27 LUFS |
| `split_sentences` | `True` | render long texts sentence by sentence (sooner first audio) |
| `word_timings` | `True` | attach estimated word timings |

`await tts.load_voice("me.wav")` encodes a reference ahead of its first use.

### Barge-in

Synthesis runs on one worker thread per provider instance. When a request is closed (the
session truncates the agent on barge-in), a forward hook on the T3 transformer stops the
generation at its next token, so the GPU is free for the next reply within milliseconds
instead of after the whole sentence.

## Performance

RTX 5070 Ti (16 GB, Blackwell, driver 615.71), Ryzen 5 5600, torch 2.14 + CUDA 13.0. First
audio and real-time factor of `synthesize()` on three short customer-service sentences
(3–5 s of audio), after `warmup()`, median of 9 runs; see the [Qwen3-TTS page](qwen-tts.md#performance)
for the full comparison with Kokoro and Pocket TTS.

| Model | Device | First audio p50 | RTF | VRAM |
|---|---|---:|---:|---:|
| Turbo | CUDA | 980 ms | 0.29 | ~3.5 GB |
| Nano | CUDA | 466 ms | 0.15 | ~2.5 GB |

Chatterbox renders a whole sentence before returning audio, so the first audio grows with
the length of the first sentence (the session's sentence adapter keeps the first chunk of
a reply short). The T3 decoding loop is launch-bound on the GPU; see the follow-ups in
the pull request (#79).
