# Smart Turn v3 (`turn_detector: smart_turn`)

[Smart Turn](https://github.com/pipecat-ai/smart-turn) v3.2 (by Daily / Pipecat) is the default local end-of-turn detector. The weights, training data and code are BSD-2-Clause. It is a Whisper-Tiny encoder with a linear head (8M parameters). It listens to the audio of the user's current turn and returns the probability that the turn is **complete**. The decision comes from prosody and other acoustic cues, not from a transcript. It covers 23 languages and runs on [ONNX Runtime](https://onnxruntime.ai/) on the CPU.

The cascade runs it at every pause the VAD reports. It commits quickly when the user sounds finished and waits longer when they paused mid-sentence.

## Setup

```bash
pip install 'voice-agent-next[smart-turn]'    # or: uv sync --extra smart-turn
```

The only dependency is `onnxruntime`: no torch, no transformers. The model is downloaded on first use; see [Model download and offline use](#model-download-and-offline-use).

## Usage

```python
from voice_agent_next import AgentSession, create
from voice_agent_next.providers.smart_turn import SmartTurnDetector

session = AgentSession(stt=..., llm=..., tts=..., vad="silero", turn_detector="smart_turn")

detector = create("turn", "smart_turn/smart-turn-v3.2-gpu")  # fp32 variant
detector = SmartTurnDetector(threshold=0.6, num_threads=2)
await detector.warmup()  # optional: download + load now, off the event loop

# standalone: the audio of the user's current turn, any rate / channel count
p = await detector.predict_end_of_turn(audio=turn_audio)  # P(turn complete)
```

```yaml
# agent.yaml
vad: silero
turn_detector: {provider: smart_turn, threshold: 0.5}
```

## How the cascade uses it

1. The VAD reports a *candidate* pause (`END_OF_SPEECH`) after `min_silence_duration` (0.25 s by default).
2. The cascade flushes the STT. It then scores the current turn with Smart Turn: every speech segment since the turn started, plus 0.5 s of audio from before the speech (`CascadeOptions.turn_audio_prefix`).
3. The next step depends on the probability `p`:
   * If `p >= threshold`, the turn is committed `min_endpointing_delay` (0.4 s) after the end of speech.
   * Otherwise the cascade waits up to `max_endpointing_delay` (2.5 s).
   * If the user speaks again before the commit, the turn continues. It is scored again, as a whole, at the next pause, as the Smart Turn authors recommend.

Keep the VAD's `min_silence_duration` short (0.2–0.25 s). Smart Turn only runs after the VAD pause, so a longer stop time adds its full length to every reply (research note 04, §8.6).

## Models

Files come from the Hugging Face repo [`pipecat-ai/smart-turn-v3`](https://huggingface.co/pipecat-ai/smart-turn-v3), pinned to revision `f766f81d3cfdf7737ac64aad813d91bbfd56bf93` (2026-01-07, the v3.2 release). Their SHA-256 checksums are pinned in `voice_agent_next/providers/smart_turn.py`.

| Model id | Weights | Size | Accuracy on the official v3.2 test set |
|---|---|---|---|
| `smart-turn-v3.2-cpu` (default) | int8 (static quantization) | 8.7 MB | 92.63 % |
| `smart-turn-v3.2-gpu` | fp32 | 32.4 MB | 93.71 % |
| `smart-turn-v3.1-cpu`, `smart-turn-v3.1-gpu`, `smart-turn-v3.0` | earlier releases, same input format | 8.7 / 32.4 / 8.8 MB | — |

The accuracy figures are from the upstream benchmark reports: 31,527 samples in 23 languages.

The `-gpu` file is simply the fp32 export, and it runs on the CPU too. It is about 1 point more accurate and roughly 30 % slower there. To run it on a GPU, install `onnxruntime-gpu` and pass `providers=["CUDAExecutionProvider", "CPUExecutionProvider"]`.

## Options

| Option | Default | Meaning |
|---|---|---|
| `model` | `"smart-turn-v3.2-cpu"` | Model id. The `smart-turn-` prefix is optional (`"v3.2-gpu"`). |
| `model_path` | `None` | Local `.onnx` file to use instead of the download, e.g. a fine-tuned model. Its file name becomes the model name in metrics. |
| `threshold` | `0.5` | Probability at or above which the turn counts as complete. This is the upstream default. Raise it to cut the user off less often, at the cost of slower replies. |
| `revision` | pinned commit | Hugging Face revision to download from. Checksums are only verified for the pinned revision. |
| `providers` | `["CPUExecutionProvider"]` | ONNX Runtime execution providers. |
| `num_threads` | `1` | ONNX Runtime intra-op threads. One thread with spin-waiting disabled keeps CPU usage predictable next to STT and TTS. Raise it on dedicated machines to cut latency (see below). |

## How it works

This is the input contract of the reference [`inference.py`](https://github.com/pipecat-ai/smart-turn/blob/main/inference.py), reproduced exactly:

* **Audio:** the turn is down-mixed to mono and resampled to 16 kHz. Longer turns keep their **last 8 s**. Shorter ones are zero-padded at the **start**, so the speech always ends the window.
* **Features:** `log_mel_features()` is a numpy port of `transformers.WhisperFeatureExtractor(chunk_length=8)` with `do_normalize=True`. It needs neither torch nor transformers at runtime.
  * The waveform is normalized to zero mean and unit variance.
  * The spectrogram uses a 400-sample periodic Hann window with a 160-sample hop, centered with reflect padding, and drops the last frame.
  * An 80-bin Slaney mel filter bank is applied, then log10.
  * Values are floored 8 decades below the maximum and scaled with `(x + 4) / 4`.
  * The result is an `(80, 800)` float32 array.
* **Model:** the input `input_features` has shape `(batch, 80, 800)` float32. The output `logits` has shape `(batch, 1)` and is already a sigmoid probability.
* **Threads:**
  * Feature extraction and inference run together in a worker thread (`asyncio.to_thread`), so the event loop never blocks.
  * The ONNX Runtime session uses 1 intra-op thread (`num_threads`) and 1 inter-op thread, sequential execution, and spin-waiting disabled.
  * One session is shared by all connections that use the detector. It loads lazily, once, even under concurrent calls, or up front via `warmup()`. `aclose()` releases it.
* **No audio:** if a turn has no audio (e.g. text-only input), the detector returns `1.0` without running the model. Silence-based endpointing then decides.

### Feature extractor equivalence

* **Golden test:** `tests/providers/data/smart_turn_log_mel.npy` holds features that `transformers` computed for three test signals. They cover a short, start-padded utterance, a 10 s chirp truncated to its last 8 s, and white noise. The numpy port matches them to 1.2e-7 (one float32 rounding step), and its mel filter bank is identical. The test tolerance is 1e-4, stricter than the 1e-3 acceptance criterion.
* **Regenerating the golden file:** the generator lives in the test module. Run it in a throwaway environment: `uv run --with transformers==5.17.0 python tests/providers/test_smart_turn.py`. It needs no torch, because upstream inference uses transformers' numpy code path.
* **End-to-end check:** we compared against the upstream pipeline (transformers features plus the same ONNX session) on 400 random samples of the official test set:
  * **fp32 model:** probabilities agree to within 4e-6.
  * **int8 model:** the median sample agrees exactly. The largest gap was 0.04 on one sample, because a one-rounding-step feature difference can flip an int8 activation-quantization step. No decision changed.

## Languages

Smart Turn v3.2 covers 23 languages: Arabic, Bengali, Chinese, Danish, Dutch, English, Finnish, French, German, Hindi, Indonesian, Italian, Japanese, Korean, Marathi, Norwegian, Polish, Portuguese, Russian, Spanish, Turkish, Ukrainian and Vietnamese (`SmartTurnDetector.languages`).

`supports_language()` accepts several spellings:

* ISO 639-1 and 639-3 codes: `"de"`, `"deu"`, `"nb"`.
* BCP-47 tags: `"pt-BR"`, `"zh-CN"`.
* English names: `"english"`.

It returns `True` for an unknown (`None`) language.

Accuracy varies by language. For `v3.2-cpu` it ranges from about 96–97 % (Korean, Turkish, German, Japanese) down to 79–84 % (Vietnamese, Marathi, Bengali). See the benchmark reports in the model repo.

## Latency and performance

Measured on an AMD Ryzen 5 5600 with onnxruntime 1.30, numpy 2.5 and Python 3.13:

| | `v3.2-cpu` (int8) | `v3.2-gpu` (fp32, on CPU) |
|---|---:|---:|
| ONNX inference, 1 thread (default) | 48 ms | 62 ms |
| ONNX inference, 4 threads | 17 ms | 22 ms |
| Feature extraction (numpy) | 4.4 ms | 4.4 ms |
| `predict_end_of_turn`, 16 kHz input, 1 thread | ≈ 65 ms | ≈ 75 ms |
| Cold start from the cache (load + first inference) | ≈ 115 ms | ≈ 135 ms |

* **Frequency:** Smart Turn runs once per VAD pause, not per audio frame.
* **Where the time goes:** the cascade flushes the STT first, then runs the detector, and commits `min_endpointing_delay` (0.4 s) after the end of speech. The VAD pause (0.25 s) and a fast STT flush therefore leave inference only partly hidden inside that wait.
* **Other rates:** 8, 24 or 48 kHz input is resampled first. For 8 s of audio this takes about 1 ms with soxr (`pip install 'voice-agent-next[resample]'`) and about 50 ms with the pure-numpy fallback.
* **Upstream figure:** the upstream 12.6 ms (AWS c7a.2xlarge) uses all cores. Use `num_threads` to trade CPU for latency.
* **Metrics:** every prediction emits `EOTMetrics`: probability, threshold, decision, and `inference_duration`, which includes the hop to the worker thread.

## Accuracy notes

* **Independent benchmark:** on LiveKit's eot-bench (English), Smart Turn v3.2 cuts the user off 35.2 % of the time at a 300 ms delay and 14.8 % at 600 ms. Silence alone scores 55.6 % and 21.7 %. So it clearly beats a VAD alone, but trails the best proprietary detectors (research note 04, §3.2).
* **Audio only:** the model ignores the dialog context, so a bare "yes" that answers a question sounds unfinished to it. Text or context-aware detectors are complementary.
* **Trained on speech:** synthetic signals get arbitrary probabilities. This includes pure tones, the amplitude-modulated tone of `providers.mock.synth_speech` and formant-synthesized vowels. Tests that need meaningful end-of-turn decisions should use `MockTurnDetector` or a fake ONNX session, as `tests/providers/test_smart_turn.py` does.
* **int8 sensitivity:** tiny input changes, such as resampling or a different audio codec, can move the int8 model's probability by a few hundredths. Identical audio always gives identical results.

## Model download and offline use

* **Download path:** models are fetched with `voice_agent_next.utils.download.hf_file`.
  * If `huggingface_hub` is installed, they go into the shared Hugging Face cache.
  * Otherwise they are saved as `<cache>/hf/pipecat-ai/smart-turn-v3/<file>` and checked against the pinned SHA-256. `<cache>` is `$VAN_CACHE_DIR` or the platform's user cache directory (`~/.cache/voice-agent-next/models` on Linux).
* **Offline:** after the first download everything works offline. With `VAN_OFFLINE=1`, a missing model raises `DownloadError` instead of reaching the network.
* **Preparing ahead of time:** to prepare a machine or image, run `python -c "import asyncio; from voice_agent_next.providers.smart_turn import SmartTurnDetector; asyncio.run(SmartTurnDetector().warmup())"`. Or copy the file and pass `model_path=...`.

## Errors

| Error | When |
|---|---|
| `ConfigurationError` | `threshold` outside `[0, 1]`, `num_threads < 1`, or a missing `model_path` file |
| `MissingDependencyError` | `onnxruntime` is not installed (`pip install 'voice-agent-next[smart-turn]'`); raised before any download |
| `DownloadError` | the model is not cached and cannot be downloaded (network failure, checksum mismatch, `VAN_OFFLINE=1`) |
| `ProviderError` | ONNX Runtime fails to load or run the model |

## Tests

```bash
uv run pytest -q tests/providers/test_smart_turn.py                                      # unit tests: offline, fake ONNX Runtime
uv sync --extra smart-turn && uv run pytest -m model tests/providers/test_smart_turn.py -s   # real models
```

The unit tests cover:

* the golden features;
* padding and truncation;
* resampling and down-mixing;
* the ONNX feed (name, shape, dtype);
* the worker thread, lazy loading and error mapping;
* the session options;
* the registry entry and languages;
* the `AgentSession(..., turn_detector="smart_turn")` wiring.

The model tests download both models and a 352 KB public-domain speech clip, then check that:

* both models score the clip's complete sentence as complete, and two versions cut mid-sentence ("ask not", "ask what you can do") as incomplete. The clip is J. F. Kennedy's 1961 inaugural address from the whisper.cpp samples, pinned by SHA-256 and shared with the Silero tests.
* 48 kHz stereo input agrees with 16 kHz input.
* the model works with `VAN_OFFLINE=1` once cached.
* `AgentSession(stt=..., llm=..., tts=..., vad="silero" | "energy", turn_detector="smart_turn")` holds the turn open after "ask not" and commits it once the sentence is finished.
