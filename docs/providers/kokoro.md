# Kokoro TTS (`kokoro`)

[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) is a small, permissively licensed
(Apache-2.0) text-to-speech model: 82M parameters, 24 kHz mono output, 54 voices in
8 languages. voice-agent-next runs it locally through
[kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) (MIT) on ONNX Runtime. It needs
no GPU and no API key, and works offline once the model files are cached.

| | |
|---|---|
| Spec | `kokoro`, `kokoro/<model>` (for example `kokoro/v1.0-fp16`) |
| Class | `voice_agent_next.providers.kokoro.KokoroTTS` |
| Extra | `voice-agent-next[kokoro]` (`kokoro-onnx>=0.6.1`, which pulls in `onnxruntime`, `phonemizer` and a bundled espeak-ng) |
| Python | 3.11 to 3.13 (see [Python 3.14](#python-314)) |
| Output | 24 kHz, mono, s16le, in 50 ms chunks |
| Streaming | audio out: per sentence. Text in: none, so `stream()` uses the `SentenceStreamAdapter` |

## Setup

```bash
pip install 'voice-agent-next[kokoro]'   # or, in a checkout: uv sync --extra kokoro
```

There are no system packages to install, because espeak-ng ships inside the
`espeakng-loader` wheel. The model files are downloaded on first use (see [Models](#models))
into the shared model cache: `$VAN_CACHE_DIR`, or `~/.cache/voice-agent-next/models` on Linux.
They land under `kokoro/model-files-v1.1/`. Every file is checked against a pinned sha256.
To prepare an offline machine, run `await tts.warmup()` once while online. After that,
`VAN_OFFLINE=1` guarantees no network access.

**GPU (optional).** Install an accelerated ONNX Runtime build in place of `onnxruntime`:
`onnxruntime-gpu` for NVIDIA CUDA (kokoro-onnx offers it as `kokoro-onnx[gpu]`), or
`onnxruntime-directml` for Windows. See [Execution providers](#execution-providers).

### Python 3.14

kokoro-onnx declares `Requires-Python <3.14`, so the `kokoro` extra (and `local`) installs
nothing on Python 3.14, and `van providers` reports the dependency as missing. Forcing the
install (`uv pip install kokoro-onnx`, which ignores the upper bound) does not help yet. In
our test on Linux, phonemizer 3.4 on Python 3.14 could not point espeak-ng at its data
directory, and espeak-ng then exited the whole process. Use Python 3.11 to 3.13.

## Usage

```python
from voice_agent_next import create

tts = create("tts", "kokoro")  # v1.0 fp32, voice af_heart
tts = create("tts", "kokoro/v1.0-fp16", voice="bf_emma", speed=1.1)
await tts.warmup()  # download + load now instead of on the first request

audio = await tts.synthesize("Hello! How can I help you today?").collect()
async for chunk in tts.synthesize("Chunks arrive as soon as a sentence is ready."):
    play(chunk.frame)  # AudioFrame, 24 kHz mono

print(await tts.list_voices())
```

In a config file:

```yaml
tts: {provider: kokoro/v1.0, voice: af_heart, speed: 1.0}
```

## Models

All files come from the kokoro-onnx
[`model-files-v1.1`](https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.1)
release. These exports report phoneme durations (kokoro-onnx uses them to place pauses), take
a float `speed`, and embed their vocabulary.

| Model id | File | Download | Voice pack | Notes |
|---|---|---|---|---|
| `v1.0` (default) | `kokoro-v1.0.onnx` | 326 MB | `voices-v1.0.bin` (28 MB, 54 voices) | fp32, reference quality |
| `v1.0-fp16` | `kokoro-v1.0.fp16.onnx` | 164 MB | same | spectral correlation 0.999 vs fp32; same CPU speed as fp32 in our test; the better choice for GPUs |
| `v1.0-int8` | `kokoro-v1.0.int8.onnx` | 114 MB | same | smallest download; correlation 0.916; **slower than real time on x86 CPUs** (see [Performance](#performance)) |
| `v1.1-zh` | `kokoro-v1.1-zh.onnx` | 326 MB | `voices-v1.1-zh.bin` (54 MB, 103 voices) | Chinese + English |
| `v1.1-zh-fp16` | `kokoro-v1.1-zh.fp16.onnx` | 164 MB | same | |
| `v1.1-zh-int8` | `kokoro-v1.1-zh.int8.onnx` | 114 MB | same | |

Aliases are accepted: `int8`, `fp16` and `fp32` mean the v1.0 variants, and file names
such as `kokoro-v1.0.int8.onnx` resolve to their ids. To use your own export or a
pre-downloaded file, pass `model_path=` and/or `voices_path=`. A custom export can carry
any `model=` label (it shows up in metrics) and uses the v1.0 voice pack unless you pass
`voices_path`.

## Voices and languages

The first letter of a voice name selects its language and, unless you set `lang`, the
espeak-ng phonemizer language. Grades are from the upstream
[VOICES.md](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md) and estimate the
quality and quantity of each voice's training data; the best voices are in bold.

| Language | `lang` (espeak-ng) | Voices (v1.0) |
|---|---|---|
| American English | `en-us` | **af_heart** (A), **af_bella** (A-), af_nicole (B-), af_aoede, af_kore, af_sarah, af_alloy, af_nova, af_sky, af_jessica, af_river; am_fenrir, am_michael, am_puck, am_echo, am_eric, am_liam, am_onyx, am_santa, am_adam |
| British English | `en-gb` | **bf_emma** (B-), bf_isabella, bf_alice, bf_lily; bm_fable, bm_george, bm_lewis, bm_daniel |
| Spanish | `es` | ef_dora; em_alex, em_santa |
| French | `fr-fr` | ff_siwis (B-) |
| Hindi | `hi` | hf_alpha, hf_beta; hm_omega, hm_psi |
| Italian | `it` | if_sara; im_nicola |
| Brazilian Portuguese | `pt-br` | pf_dora; pm_alex, pm_santa |
| Japanese | `ja` | jf_alpha, jf_gongitsune, jf_nezumi, jf_tebukuro; jm_kumo |
| Mandarin Chinese | `cmn` | zf_xiaobei, zf_xiaoni, zf_xiaoxiao, zf_xiaoyi; zm_yunjian, zm_yunxi, zm_yunxia, zm_yunyang |

The v1.1-zh pack has 100 Chinese voices (`zf_001`..., `zm_009`...; the default is `zf_001`)
plus `af_maple`, `af_sol` and `bf_vale`.

Phonemization uses espeak-ng. It works well for English and the European languages, but
Japanese and Chinese through espeak-ng sound poor; upstream uses
[misaki](https://github.com/hexgrad/misaki) for those. To use a better G2P, pass
`g2p=lambda text, lang: phonemes`, and kokoro-onnx will receive phonemes instead of text.

## Options

| Option | Default | Description |
|---|---|---|
| `model` | `"v1.0"` | model id (see [Models](#models)) |
| `voice` | `"af_heart"` (`"zf_001"` for v1.1-zh) | default voice; `synthesize(text, voice=...)` overrides it per request |
| `speed` | `1.0` | speaking rate, 0.5 to 2.0 |
| `lang` | from the voice | espeak-ng language code, e.g. `en-us`, `en-gb`, `fr-fr`, `pt-br`, `cmn` |
| `model_path`, `voices_path` | downloaded | local files instead of the pinned downloads |
| `providers` | auto | ONNX Runtime execution providers, e.g. `["CPUExecutionProvider"]` or `[("CUDAExecutionProvider", {"device_id": 1}), "CPUExecutionProvider"]` |
| `num_threads` | ORT default (one per physical core) | ONNX Runtime intra-op threads; lower it when STT/LLM share the CPU |
| `chunk_duration` | `0.05` | seconds of audio per emitted chunk |
| `split_sentences` | `True` | synthesize long texts sentence by sentence: sentences shorter than 20 characters are merged, and run-ons longer than 300 are split at clauses |
| `sentence_pause`, `clause_pause` | `0.25`, `0.1` | silence (s) after `.!?` / `,;:`, inside the text and at the end of each sentence |
| `trim` | `True` | trim the silence the model generates around each sentence |
| `g2p` | `None` | custom `(text, lang) -> phonemes` function |
| `clean_text` | `True` | strip markdown and emoji before synthesis |

## How it works

* **Threading.** The model is loaded lazily (or by `warmup()`), and all phonemization and
  inference run on one dedicated worker thread per `KokoroTTS` instance. The event loop is
  never blocked, requests run one at a time in order (no CPU oversubscription from
  concurrent inferences), and the model is loaded exactly once.
* **Early audio.** kokoro-onnx's `create_stream()` works on batches of up to 510 phonemes
  and phonemizes on the event loop thread, so the provider does not use it. Instead, each
  request is split into sentences and each sentence is one inference pass, so the first
  audio is ready after the first sentence instead of after the whole text. A sentence's
  audio is emitted in `chunk_duration` chunks as soon as it is synthesized. In an agent,
  `stream()` goes through the `SentenceStreamAdapter`, which already hands over one
  sentence at a time (with a short first segment) and prefetches the next one.
* **Pauses.** kokoro-onnx trims the model's leading and trailing silence and adds pauses
  inside the text. The provider appends `sentence_pause` / `clause_pause` after each sentence,
  so consecutive sentences, which are synthesized separately, don't run together.
* **Cancellation.** Closing a stream stops it before the next sentence. A sentence that is
  already running finishes in the background and its audio is dropped.
* **Errors.** A missing extra raises `MissingDependencyError`. An unknown model, voice,
  speed or local path raises `ConfigurationError`. Download failures raise `DownloadError`,
  and ONNX Runtime or phonemizer failures raise `ProviderError`. Text with nothing to
  pronounce, such as emoji or bare punctuation, produces no audio instead of an error.

## Execution providers

By default the first available of `CUDAExecutionProvider`, `CoreMLExecutionProvider` and
`DmlExecutionProvider` is used, with `CPUExecutionProvider` as the fallback for unsupported
operators. If the accelerated session cannot be created, the provider logs a warning and
retries on CPU. Providers passed explicitly (`providers=`) are used as given. The standard
`onnxruntime` wheel is CPU-only on Linux and Windows (plus `AzureExecutionProvider`, which is
ignored) and includes CoreML on macOS.

The int8 models are dynamically quantized (`ConvInteger`), so on GPUs they gain little. Use
fp32 or fp16 there.

## Performance

Measured on an AMD Ryzen 5 5600 (6 cores / 12 threads), CPU only, ONNX Runtime 1.30,
kokoro-onnx 0.6.1, Python 3.13, default threads, voice `af_heart`. Each figure is the median
of 3 to 8 warm runs. The machine was shared with other jobs (load average 2 to 4), so treat the
numbers as indicative. TTFB is the time from the request to the first audio chunk. RTF is
synthesis time divided by audio duration; below 1 is faster than real time.

| Model | Request | Audio | TTFB | RTF |
|---|---|---|---|---|
| `v1.0` (fp32) | 27-char sentence | 2.0 s | 324 ms | 0.158 |
| `v1.0` (fp32) | 56-char sentence (one segment) | 4.2 s | 659 ms | 0.157 |
| `v1.0` (fp32) | 213-char paragraph (3 sentences) | 12.7 s | 678 ms | 0.162 |
| `v1.0` (fp32), `num_threads=4` | 56-char sentence | 4.2 s | 806 ms | 0.192 |
| `v1.0-fp16` | 56-char sentence (interleaved with fp32: 0.161) | 4.2 s | 692 ms | 0.165 |
| `v1.0-int8` | 27-char sentence | 2.0 s | 3402 ms | 1.67 |
| `v1.0-int8` | 56-char sentence | 4.2 s | 5741 ms | 1.36 |

* **In an agent**, measured with `stream()` and a 146-character reply fed word by word:
  the first audio came about 205 ms after the first text, because the adapter releases
  "Sure!" as its own segment. All 9.5 s of speech were ready after 1.7 to 2.0 s.
* **Cold start.** Downloading the 326 MB `v1.0` model, loading it and warming up took
  9.1 s. Loading from the cache, the first request took 1.9 s. Call `warmup()` at startup.
* **Why int8 is slow.** Profiling shows 92% of its time in `ConvInteger`, which has no
  optimized x86 kernel in ONNX Runtime. Use int8 only when download size matters more than
  latency.
* TTFB grows with the length of the first sentence, since Kokoro synthesizes a whole
  sentence per pass. Upstream notes that voices sound best at 100 to 200 phonemes and
  weaker on very short inputs, so the provider merges fragments under 20 characters.
* On a Raspberry Pi 4, Kokoro runs slower than real time (RTF about 3.2 with 4 threads,
  research note 03 §6). Prefer a desktop CPU or a GPU for conversational use.

Reproduce with the real-model test. It uses `v1.0-int8` by default; select another model
with `VAN_KOKORO_TEST_MODEL`:

```bash
uv sync --extra kokoro
VAN_KOKORO_TEST_MODEL=v1.0 uv run pytest -m model tests/providers/test_kokoro.py -s
```

## Limitations and follow-ups

* No word timestamps yet. The v1.1 exports report phoneme durations
  (`Kokoro.create_timed`), which could provide word timings for exact truncation.
* No voice blending (for example `af_heart:0.7,af_bella:0.3`) yet.
* espeak-ng G2P only; misaki is not bundled (use `g2p=`).
* The CoreML and DirectML paths are untested on real hardware. Upstream kokoro-onnx uses
  CPU on macOS unless told otherwise; pass `providers=["CPUExecutionProvider"]` if CoreML
  turns out slower for Kokoro's dynamic shapes.
