# Silero VAD (`vad: silero`)

[Silero VAD](https://github.com/snakers4/silero-vad) v6 is the default voice activity detector for production. It is a small (2.3 MB) MIT-licensed neural network. It runs on [ONNX Runtime](https://onnxruntime.ai/) on one CPU thread in about 0.1 ms per 32 ms window.

The provider runs the official ONNX file directly. It does **not** use the `silero-vad` pip package, because that package pulls in torch.

## Setup

```bash
pip install 'voice-agent-next[silero]'      # or: uv sync --extra silero
```

The only dependency is `onnxruntime`. The model is downloaded on first use; see [Model download and offline use](#model-download-and-offline-use).

## Usage

```python
from voice_agent_next import AgentSession, create
from voice_agent_next.providers.silero import SileroVAD

vad = create("vad", "silero")  # default model and options
vad = SileroVAD(min_silence_duration=0.3)  # override single VADOptions fields
await vad.warmup()  # optional: download + load now, off the event loop

session = AgentSession(stt=..., llm=..., tts=..., vad="silero")
```

```yaml
# agent.yaml
vad: {provider: silero, sample_rate: 16000, min_silence_duration: 0.25}
```

## Models

| Model id | File | Rates | Size | Source |
|---|---|---|---|---|
| `v6.2` (default) | `silero_vad.onnx`, ONNX opset 16 | 8 kHz, 16 kHz | 2.3 MB | [snakers4/silero-vad `v6.2`](https://github.com/snakers4/silero-vad/blob/v6.2/src/silero_vad/data/silero_vad.onnx), unchanged through v6.2.3 |

The URL and SHA-256 (`1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3`) are pinned in `voice_agent_next/providers/silero.py`. A download that does not match the checksum is rejected.

## Options

| Option | Default | Meaning |
|---|---|---|
| `model` | `"v6.2"` | Model id. With `model_path` it is only a label. |
| `sample_rate` | `16000` | Model rate: `16000` (512-sample windows) or `8000` (256-sample windows). Input audio at any rate and channel count is converted by the stream. |
| `model_path` | `None` | Local Silero v5/v6 ONNX file to use instead of the download. |
| `device` | `"cpu"` | ONNX Runtime device: `"cpu"`, `"auto"` (every available execution provider), `"cuda"`, `"coreml"`... or a list of execution providers. Keep the CPU: the model is tiny and runs with batch size 1, so a GPU only adds transfer overhead. (`force_cpu=True`/`False` is deprecated.) |
| `options` | `VADOptions()` | Thresholds and durations, see below. Single fields can also be passed as keyword arguments (`min_silence_duration=0.3`); they are applied on top of `options`. |

`VADOptions` fields (shared by all VADs, see `voice_agent_next/vad.py`):

| Field | Default | Meaning |
|---|---|---|
| `activation_threshold` | `0.5` | Probability at or above which a window counts as speech. Silero's own default. |
| `deactivation_threshold` | `activation_threshold - 0.15` | Probability below which a window counts as silence. Silero's own default. |
| `min_speech_duration` | `0.1` s | Speech needed before `START_OF_SPEECH`. With 32 ms windows this is 4 windows (0.128 s). |
| `min_silence_duration` | `0.25` s | Silence needed before `END_OF_SPEECH`. With 32 ms windows this is 8 windows (0.256 s). Short on purpose: engines treat it as a *candidate* pause and let the turn detector decide. |
| `prefix_padding_duration` | `0.5` s | Audio kept before the speech start and included in events. |
| `max_buffered_speech` | `60` s | Cap on the audio kept for one utterance. |
| `smoothing` | `0` | Exponential smoothing of probabilities (0 = off). |

## How it works

* Each window is prefixed with the last 64 samples (16 kHz) or 32 samples (8 kHz) of the previous window. A `(2, 1, 128)` recurrent state carries over from window to window. This is exactly what the reference `OnnxWrapper` does; a real-model test checks that the probabilities match it window by window.
* Model inputs are `input` (`[1, context + window]` float32), `state` (`[2, 1, 128]` float32) and `sr` (int64 scalar). Outputs are the speech probability (`[1, 1]`) and the next state.
* Every stream (`vad.stream()`) has its own state and context; `stream.reset()` clears both. All streams of one `SileroVAD` share one ONNX Runtime session, configured with 1 intra-op thread, 1 inter-op thread and spin-waiting disabled (by default idle ONNX Runtime threads spin and burn CPU).
* The model is downloaded and loaded lazily: `await vad.warmup()` does it in a worker thread; otherwise the first `vad.stream()` does it synchronously.

## Model download and offline use

* The model is cached as `<cache>/silero/silero_vad_v6.2.onnx`. `<cache>` is `$VAN_CACHE_DIR` or the platform's user cache directory (`~/.cache/voice-agent-next/models` on Linux). Print yours with `python -c "from voice_agent_next.utils.download import cache_dir; print(cache_dir())"`.
* After the first download everything works offline. With `VAN_OFFLINE=1`, a missing model raises `DownloadError` instead of reaching the network.
* To prepare an image or machine ahead of time, run `python -c "import asyncio; from voice_agent_next.providers.silero import SileroVAD; asyncio.run(SileroVAD().warmup())"`. You can also copy the file and pass `model_path=...`.

## Latency and performance

* **Granularity:** decisions are made every 32 ms. `START_OF_SPEECH` fires about `min_speech_duration` after speech begins. `END_OF_SPEECH` fires `min_silence_duration` after it ends, and its `audio_time` minus `silence_duration` is where speech ended.
* **Inference:** 0.104 ms per 512-sample window at 16 kHz and 0.086 ms per 256-sample window at 8 kHz. Measured on an AMD Ryzen 5 5600 with onnxruntime 1.30 and Python 3.13, on one thread. That is about 0.3% of one core per stream, so inference runs inline on the event loop. The `model` tests print this number on your machine.
* **Cold start:** loading the model from the cache and running one window takes about 60–90 ms. The first download is 2.3 MB.
* The VAD emits `VADMetrics` (inference count and total inference time) for each stream every 5 s of audio and when the stream closes.

## Accuracy notes

* Silero looks for *speech*, not energy. It ignores silence, white and brown noise, pure tones, and the amplitude-modulated tone that `voice_agent_next.providers.mock.synth_speech` generates. Use the `energy` VAD for tests built on synthetic signals.
* Silero lists voice-like music and very high-pitched voices as known weak spots of v6. v6.2 improved child voices, muted speech and low-quality phone calls.
* Telephony audio (8 kHz) can use `sample_rate=8000` directly and skip resampling. Both rates run the same model.

## Platform notes

* `onnxruntime` 1.24 and later ship no Intel-Mac wheels and need macOS 14 or newer on Apple Silicon. pip falls back to the newest compatible release by itself. A lock file resolved elsewhere may not, so pin `onnxruntime<1.24` for those machines.
* No torch and no GPU are needed. The ONNX file is platform-independent.

## Errors

| Error | When |
|---|---|
| `ConfigurationError` | unsupported `sample_rate`, unknown model id or option, missing `model_path` file, or a file that is not a Silero v5/v6 model |
| `MissingDependencyError` | `onnxruntime` is not installed (`pip install 'voice-agent-next[silero]'`) |
| `DownloadError` | the model is not cached and cannot be downloaded (network failure, checksum mismatch, `VAN_OFFLINE=1`) |
| `ProviderError` | ONNX Runtime fails to load or run the model |

## Tests

```bash
uv run pytest -q tests/providers/test_silero.py                       # unit tests: fake ONNX Runtime, offline
uv sync --extra silero && uv run pytest -m model tests/providers/test_silero.py -s   # real model
```

The model tests download the model and a 350 KB public-domain speech clip once (J. F. Kennedy's 1961 inaugural address, from the whisper.cpp samples, pinned by SHA-256). They check that the provider:

* detects the clip's four phrases at 8 and 16 kHz, from 16 and 48 kHz input;
* rejects silence, noise and tones;
* matches the reference `OnnxWrapper` window by window;
* works with `VAN_OFFLINE=1` once cached.
