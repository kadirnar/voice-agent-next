# faster-whisper (local STT)

[faster-whisper](https://github.com/SYSTRAN/faster-whisper) runs OpenAI's Whisper models on
[CTranslate2](https://github.com/OpenNMT/CTranslate2), locally, on the CPU (int8) or an
NVIDIA GPU (float16). Multilingual models cover 99 languages. It is the default local
multilingual recognizer of voice-agent-next.

| | |
|---|---|
| Spec | `faster_whisper/<model>` (alias `whisper/<model>`), default model `large-v3-turbo` |
| Class | `voice_agent_next.providers.faster_whisper.FasterWhisperSTT` |
| Extra | `pip install 'voice-agent-next[faster-whisper]'` (or `uv sync --extra faster-whisper`) |
| Credentials | none (models are public on the Hugging Face Hub) |
| Capabilities | batch only: no interim results; word timestamps (opt-in); language detection (multilingual models) |
| Platforms | Linux, Windows (CPU, CUDA); macOS (CPU) |

## Usage

```python
from voice_agent_next import create

stt = create("stt", "faster_whisper/small", language="en")
await stt.warmup()                 # download (first run) + load + warm-up inference
transcript = await stt.transcribe(frame)  # any sample rate / channel count
print(transcript.text, transcript.language, transcript.confidence)
```

Whisper transcribes complete utterances. In a cascade it needs a VAD: `CascadeEngine`
automatically wraps batch recognizers in `voice_agent_next.stt.StreamAdapter`, which cuts
the input into utterances at VAD end-of-speech and transcribes each one (one
`FINAL_TRANSCRIPT` per utterance, plus `START_OF_SPEECH` / `END_OF_SPEECH` from the VAD).

```yaml
# agent.yaml
stt: {provider: faster_whisper/small, language: en}
vad: energy          # use a neural VAD (Silero) for real microphones
llm: ...
tts: ...
```

The model is loaded once, lazily, in a worker thread (guarded by a lock), and each
transcription runs in `asyncio.to_thread`, so the event loop never blocks. Call
`warmup()` (the cascade's `warmup()` does it for you) so the first user turn does not pay for
the download, the load and kernel initialization.

## Models

Any model faster-whisper knows by name, any CTranslate2 Whisper repository on the Hub
(`faster_whisper/deepdml/faster-whisper-large-v3-turbo-ct2`), or a local directory holding a
converted model (`model="/models/whisper-ct2"`). Downloads go to the Hugging Face cache
(`~/.cache/huggingface/hub`, or `download_root=`).

| Model | Download | Languages | Notes |
|---|---|---|---|
| `tiny` / `tiny.en` | 78 MB | 99 / English | fastest, least accurate |
| `base` / `base.en` | 148 MB | 99 / English | good CPU default for English |
| `small` / `small.en` | 486 MB | 99 / English | best accuracy that is still interactive on a desktop CPU |
| `medium` / `medium.en` | 1.5 GB | 99 / English | GPU recommended |
| `large-v3` | 3.1 GB | 99 | most accurate Whisper; GPU |
| `large-v3-turbo` (`turbo`) — **default** | 1.6 GB | 99 | large-v3 encoder with a 4-layer decoder: close to large-v3 accuracy, much faster; GPU recommended |
| `distil-large-v3.5` | 1.5 GB | English | distilled large-v3 |

`*.en` and `distil-*` models are English-only (`capabilities.language_detection` is `False`).

## Options

| Option | Default | Meaning |
|---|---|---|
| `model` | `large-v3-turbo` | model name, Hub repository or local directory |
| `language` | `None` | language code (`"en"`, `"de"`...; `"en-US"` becomes `"en"`); `None` detects it per utterance |
| `device` | `"auto"` | `"auto"`, `"cpu"` or `"cuda"` (see below) |
| `device_index` | `0` | CUDA device id, or a list of ids to spread concurrent requests |
| `compute_type` | `"auto"` | `float16` on CUDA, `int8` on CPU; or any [CTranslate2 type](https://opennmt.net/CTranslate2/quantization.html) (`int8_float16`, `float32`...) |
| `beam_size` | `1` | greedy decoding for the lowest latency (Whisper's own default is 5) |
| `word_timestamps` | `False` | fill `Transcript.words` (word, start, end, probability) |
| `vad_filter` | `False` | faster-whisper's own Silero VAD pass; off because the pipeline already segments speech |
| `initial_prompt` | `None` | text that primes the decoder (spelling, style, vocabulary) |
| `hotwords` | `None` | hint phrases (names, product terms) |
| `cpu_threads` | `0` | CTranslate2 threads on CPU (0 = its default of 4, or `OMP_NUM_THREADS`) |
| `num_workers` | `1` | model replicas that can transcribe concurrently (several sessions sharing one instance) |
| `download_root` | `None` | model cache directory (default: the Hugging Face cache) |
| `local_files_only` | `False` | never download; `VAN_OFFLINE=1` (and `HF_HUB_OFFLINE=1`) imply it |
| `transcribe_options` | `{}` | extra `WhisperModel.transcribe()` arguments (`temperature`, `no_speech_threshold`, `task`...), applied last |

`Transcript.confidence` is `exp(mean token log-probability)` over the utterance, and
`start_time` / `end_time` are relative to the start of the utterance audio.

## Devices and compute types

`device="auto"` uses CUDA only when CTranslate2 reports a CUDA device **and** a warm-up
inference on it succeeds; otherwise it logs a warning and loads the model on the CPU. The
check matters because a visible GPU is not enough: CTranslate2's pip wheels (Linux and
Windows) link against **cuBLAS 12 and cuDNN 9** at run time and do not ship them. Without
them the model loads on the GPU and then fails on the first inference with
`Library libcublas.so.12 is not found or cannot be loaded`. Forcing `device="cuda"` turns
that into a `ProviderError` during `warmup()` instead of a silent fallback.
`resolved_device` and `resolved_compute_type` tell you what was picked.

To enable the GPU on Linux, install the NVIDIA runtime wheels and put them on the library path
before starting Python:

```bash
pip install nvidia-cublas-cu12 'nvidia-cudnn-cu12==9.*'
export LD_LIBRARY_PATH=$(python -c 'import os, nvidia.cublas.lib, nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))')
```

A system CUDA 12 toolkit with cuDNN 9 works too. On Windows, put the cuBLAS 12 and cuDNN 9
DLLs on `PATH`. macOS wheels are CPU-only (int8 runs well on Apple Silicon); for Metal use
an MLX provider.

GPUs newer than the CTranslate2 build (for example the RTX 50xx series) run through the
driver's PTX JIT: the very first CUDA inference on such a machine took about 11 s here while
kernels were compiled, and about 0.25 s afterwards (the driver caches them in
`~/.nv/ComputeCache`). `warmup()` absorbs this cost.

`compute_type="auto"` picks the first type CTranslate2 supports on the device: `float16`,
`int8`, `float32` on CUDA; `int8`, `float32` on CPU. An explicit type is used as given, and a
type the device cannot run raises `ConfigurationError`.

## Latency

Whisper transcribes after the utterance ends (VAD end-of-speech), so the time to the final
transcript is the transcription time below. Some consequences:

* **Short utterances cost almost as much as long ones.** Whisper always encodes a 30 s
  window, so on the CPU below a 2.5 s utterance took 78-91 % of the time of an 11 s one.
  Per-utterance latency matters more than RTF for turn-taking.
* **Set `language` when you know it.** With `language=None`, multilingual models encode the
  audio twice: once to detect the language and once to transcribe (faster-whisper 1.2.1
  does not reuse the first pass). That made `base` and `small` about 2x slower on the CPU
  below.
* `beam_size=1` (the default) is the fastest; `word_timestamps=True` adds an alignment pass.
* In a local stack the LLM and TTS compete for the same CPU cores; cap them with
  `cpu_threads`.

Measured with `FasterWhisperSTT` defaults (`beam_size=1`, no word timestamps) on the 11 s
public-domain JFK clip (RTF = transcription time / audio duration) and on its first 2.5 s
(one typical voice-agent turn); median of 5 runs after `warmup()`, faster-whisper 1.2.1,
CTranslate2 4.8.2, 2026-09-24.

**CPU:** AMD Ryzen 5 5600 (6 cores / 12 threads), `int8`, CTranslate2's default 4 threads. The
machine was busy with other jobs (load average ~7-10), so treat these as upper bounds.

| Model | `language` | 11 s clip | RTF | 2.5 s utterance |
|---|---|---|---|---|
| `tiny.en` | (English-only) | 0.23-0.32 s | 0.021-0.029 | 200-245 ms |
| `base.en` | (English-only) | 0.52-0.55 s | 0.047-0.050 | 430-480 ms |
| `tiny` | `"en"` / detect | 0.31 / 0.45 s | 0.028 / 0.041 | 245 / 365 ms |
| `base` | `"en"` / detect | 0.45 / 0.88 s | 0.041 / 0.080 | 356 / 778 ms |
| `small` | `"en"` / detect | 1.22 / 2.60 s | 0.111 / 0.237 | 1,018 / 2,360 ms |
| `large-v3-turbo` | — | not measured | — | — |

**GPU:** NVIDIA RTX 5070 Ti (16 GB), `float16`, with cuBLAS 12 / cuDNN 9 on the library path,
`language="en"`.

| Model | 11 s clip | RTF | 2.5 s utterance |
|---|---|---|---|
| `tiny.en` | 0.041 s | 0.004 | 19 ms |
| `tiny` | 0.063 s | 0.006 | 35 ms |
| `base` | 0.057 s | 0.005 | 42 ms |
| `small` | 0.108 s | 0.010 | 52 ms |
| `large-v3-turbo` | not measured | — | — |

`large-v3-turbo` was not measured: its 1.6 GB download is over the dev machine's download
budget. Its encoder is the large-v3 one, roughly 5-7x the compute of `small`'s, so on a
CPU like this one expect several seconds per utterance: use `base`/`small` (with `language`
set) for CPU agents and keep `turbo` for GPUs.

Reproduce the table with a loop like this (`clip` is the 11 s JFK recording as an
`AudioFrame`):

```python
stt = FasterWhisperSTT(model="base", device="cpu", language="en")
await stt.warmup()
t0 = time.perf_counter(); await stt.transcribe(clip); rtf = (time.perf_counter() - t0) / clip.duration
```

## Errors

| Situation | Exception |
|---|---|
| `faster-whisper` not installed | `MissingDependencyError` (at construction, with the install command) |
| unknown model / revision, bad `device`, `compute_type`, language or option | `ConfigurationError` |
| gated or private repository, HTTP 401/403 | `AuthenticationError` (log in with `hf auth login` or set `HF_TOKEN`) |
| HTTP 429 from the Hub | `RateLimitError` |
| network failure, or model not cached while offline | `ProviderConnectionError` |
| CTranslate2 failure (CUDA out of memory, missing CUDA libraries with `device="cuda"`...) | `ProviderError` |

A failed load is retried on the next call.

## Limitations

* No partial transcripts: the final transcript arrives only after the VAD reports the end of
  speech.
* Whisper can hallucinate short phrases ("Thank you.") on noise that the VAD let through.
  Use a neural VAD, and tune `transcribe_options` (`no_speech_threshold`,
  `log_prob_threshold`) or enable `vad_filter` if it happens.
* A transcription that is already running finishes in its worker thread even when the
  turn is cancelled (its result is discarded).
