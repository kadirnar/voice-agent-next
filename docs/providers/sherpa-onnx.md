# sherpa-onnx (local streaming STT, TTS and VAD)

[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) (k2-fsa, Apache-2.0) runs speech
recognition, synthesis and voice activity detection on ONNX Runtime, on every desktop OS
and on mobile-class hardware. Its main draw for voice agents is **truly streaming
recognition**: transducer models decode the audio in 80–1,000 ms chunks while the user is
speaking, so the final transcript is ready a few tens of milliseconds after the end of
speech instead of after a full re-transcription (faster-whisper `base` needs ≈ 360 ms on a
desktop CPU for the same step).

| | |
|---|---|
| Specs | `sherpa-onnx/<model>` (alias `sherpa/<model>`) for `stt`, `tts` and `vad` |
| Classes | `voice_agent_next.providers.sherpa_onnx.SherpaOnnxSTT`, `SherpaOnnxTTS`, `SherpaOnnxVAD` |
| Extra | `pip install 'voice-agent-next[sherpa-onnx]'` (or `uv sync --extra sherpa-onnx`) |
| Defaults | STT `nemo-fastconformer-en-80ms`, TTS `piper-en_US-libritts_r-medium`, VAD `silero` |
| Credentials | none; models are public GitHub release assets |
| Platforms | CPU wheels (ONNX Runtime bundled) for Python 3.11–3.14 on Linux x86-64/aarch64/armv7l, macOS x86-64/arm64, Windows x86-64/arm64/x86 |

## Setup

```bash
pip install 'voice-agent-next[sherpa-onnx]'
```

The `sherpa-onnx` wheel depends on `sherpa-onnx-core` (the native libraries, same
version); the extra pins both, because uv's universal lock otherwise misses the core
package. If `import sherpa_onnx` fails with a missing `libonnxruntime`/`libsherpa-onnx`,
the core package is missing or has another version.

Models are downloaded on first use (`warmup()` does it ahead of time) into the shared model
cache — `$VAN_CACHE_DIR`, or `~/.cache/voice-agent-next/models` on Linux — under
`sherpa-onnx/`. Every catalog asset is pinned to a sha256 and release archives are
extracted with path-traversal protection (absolute paths, `..`, links out of the archive
and device files are refused) into a temporary directory that is moved into place
atomically. After the first run, `VAN_OFFLINE=1` guarantees no network access.

## Usage

```python
from voice_agent_next import create

stt = create("stt", "sherpa-onnx/nemo-fastconformer-en-80ms")
tts = create("tts", "sherpa-onnx/kokoro-multi-lang-v1_0-int8", voice="af_heart")
vad = create("vad", "sherpa-onnx/silero")
await stt.warmup()  # download (first run) + load + measure the chunk geometry
```

```yaml
# agent.yaml: a fully local cascade
stt: {provider: sherpa-onnx/nemo-fastconformer-en-80ms, language: en}
vad: silero                 # or sherpa-onnx/silero (no onnxruntime package needed)
turn_detector: smart_turn
llm: {provider: "ollama/LiquidAI/lfm2.5-1.2b-instruct:latest"}
tts: {provider: sherpa-onnx/kokoro-multi-lang-v1_0-int8, voice: af_heart}
```

Every sherpa-onnx call of a component runs on one dedicated worker thread (sherpa objects
are never touched from two threads, and the event loop never blocks).

## Speech recognition

### Streaming models (`OnlineRecognizer`)

Zipformer, NeMo FastConformer and Nemotron transducers (plus Paraformer and CTC models from
a local directory). `capabilities.streaming` and `interim_results` are `True`, so the
cascade uses the recognizer directly (no `StreamAdapter`):

* audio is decoded as it arrives and every change of the hypothesis is emitted as
  `INTERIM_TRANSCRIPT` (preceded by `START_OF_SPEECH` for the first words);
* `flush()` — sent by the cascade when its VAD / turn detector ends the user's turn —
  pads the undecoded tail with *just enough* silence for the model's last chunk (computed
  from the chunk geometry measured at load time; upstream's fixed 0.66 s costs up to two
  extra decoding steps), decodes it and emits `FINAL_TRANSCRIPT` + `END_OF_SPEECH`. The
  next audio goes into a fresh recognizer stream, and word timings stay on the input clock;
* sherpa's own endpoint rules are **off** by default (the cascade decides when a turn
  ends). With `endpoint_detection=True` (rules `rule1_min_trailing_silence`,
  `rule2_min_trailing_silence`, `rule3_min_utterance_length`) an endpoint emits
  `FINAL_TRANSCRIPT` + `END_OF_SPEECH` by itself, for use without a VAD.

`transcribe()` works with streaming models too (one pass over the whole clip).

### Offline models (`OfflineRecognizer`)

Moonshine, Parakeet TDT, SenseVoice and Whisper transcribe whole utterances
(`capabilities.streaming=False`). In a cascade they need a VAD: `CascadeEngine` wraps them in
`StreamAdapter`, like faster-whisper. Audio longer than the model handles in one pass
(8 s for Moonshine, 28 s for SenseVoice, 29 s for Whisper) is split at the quietest point
of a pause. SenseVoice and Parakeet v3 report the detected language.

## Models

`SHERPA_MODELS` is the pinned catalog. The release archive name
(`sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06`) works as a model name too. Sizes
are download sizes.

### Speech recognition

| Model | Type | Size | Languages | License | Notes |
|---|---|---:|---|---|---|
| `nemo-fastconformer-en-80ms` — **default** | streaming | 103 MB | en | CC-BY-4.0 | NVIDIA FastConformer hybrid large (114M), 80 ms look-ahead, int8, lowercase, no punctuation |
| `nemo-fastconformer-en-480ms` | streaming | 106 MB | en | CC-BY-4.0 | 480 ms look-ahead: more accurate, later interims |
| `nemo-fastconformer-en-1040ms` | streaming | 104 MB | en | CC-BY-4.0 | 1,040 ms look-ahead |
| `zipformer-en-kroko` | streaming | 57 MB | en | CC-BY-SA-4.0 | Banafo Kroko community Zipformer: cased and punctuated |
| `zipformer-fr-kroko` / `-de-kroko` | streaming | 57 / 58 MB | fr / de | CC-BY-SA-4.0 | as above |
| `zipformer-es-kroko` | streaming | 124 MB | es | CC-BY-SA-4.0 | as above |
| `nemotron-en-{80,160,560,1120}ms` | streaming | 464 MB | en | NVIDIA Open Model License | Nemotron Speech Streaming 0.6B, cache-aware, int8; cased and punctuated |
| `nemotron-3.5-{80,160,320,560,1120}ms` | streaming | 475 MB | 40 locales | OpenMDW-1.1 | Nemotron 3.5 ASR Streaming 0.6B; `language=` is the stream's language prompt, `"auto"` detects |
| `moonshine-tiny-en` | offline | 30 MB | en | MIT | Moonshine v2 tiny (quantized), edge-class; cased and punctuated |
| `moonshine-base-en` | offline | 111 MB | en | MIT | Moonshine v2 base (quantized) |
| `parakeet-tdt-0.6b-v2` | offline | 482 MB | en | CC-BY-4.0 | NVIDIA Parakeet TDT 0.6B v2 (int8); cased and punctuated |
| `parakeet-tdt-0.6b-v3` | offline | 487 MB | 25 European languages | CC-BY-4.0 | Parakeet TDT 0.6B v3 (int8); language auto-detected |
| `sense-voice` | offline | 166 MB | zh, en, ja, ko, yue | FunASR model license | SenseVoice Small (int8): language ID, inverse text normalization |

Whisper (`kind="offline-whisper"`), Paraformer, CTC and any other sherpa-onnx model work
from a local directory or an archive URL (see [Other models](#other-models)).

### Speech synthesis

| Model | Size | Languages | License | Notes |
|---|---:|---|---|---|
| `piper-en_US-libritts_r-medium` — **default** | 23 MB | en-US | MIT (voice), CC-BY-4.0 (LibriTTS-R data) | Piper VITS, 904 speakers (`voice="0"`…), 22.05 kHz |
| `piper-en_US-ljspeech-medium` | 21 MB | en-US | MIT (voice), public domain (LJSpeech) | Piper VITS, one speaker, 22.05 kHz |
| `kokoro-multi-lang-v1_0` | 350 MB | en, zh, es, fr, hi, it, ja, pt | Apache-2.0 | Kokoro-82M v1.0 (fp32), 54 voices by name (`af_heart`, `am_adam`…), 24 kHz |
| `kokoro-multi-lang-v1_0-int8` | 132 MB | en, zh, es, fr, hi, it, ja, pt | Apache-2.0 | Kokoro-82M v1.0 (int8), 54 voices, 24 kHz |
| `matcha-en_US-ljspeech` | 131 MB | en | Apache-2.0 (model), MIT (Vocos), public domain (LJSpeech) | Matcha-TTS + Vocos 22 kHz vocoder (downloaded separately) |

### Voice activity detection

| Model | Size | License | Notes |
|---|---:|---|---|
| `silero` — **default** | 0.6 MB | MIT | Silero VAD, 32 ms windows |
| `ten-vad` | 0.3 MB | Apache-2.0 with additional conditions | TEN VAD, 16 ms windows |

sherpa's VAD answers speech / non-speech per window, so the probability seen by
`VADStream` is 1.0 or 0.0: `activation_threshold` is applied inside sherpa (with its fixed
hysteresis) and `smoothing` has no useful effect. Minimum speech/silence durations and
padding (`VADOptions`) work as usual.

### Other models

```python
create("stt", "sherpa-onnx", model="/models/sherpa-onnx-whisper-tiny.en")  # kind guessed
create("stt", "sherpa-onnx", model="https://…/model.tar.bz2", sha256="…", kind="online-transducer")
create(
    "stt",
    "sherpa-onnx",
    model="/models/x",
    kind="offline-nemo-ctc",
    files={"model": "/models/x/model.int8.onnx", "tokens": "/models/x/tokens.txt"},
)
```

The kind (`ModelKind`: `online-transducer`, `online-paraformer`, `online-zipformer2-ctc`,
`online-nemo-ctc`, `offline-transducer`, `offline-nemo-transducer`, `offline-nemo-ctc`,
`offline-moonshine`, `offline-moonshine-v2`, `offline-sense-voice`, `offline-whisper`,
`tts-vits`, `tts-kokoro`, `tts-matcha`, `vad-silero`, `vad-ten`) is guessed from the file
and directory names of a local directory (sherpa-onnx naming); pass `kind=` otherwise.
Files are located by role (`*.int8.onnx` preferred) and `files={role: path}` overrides any
of them.

## Options

### `SherpaOnnxSTT`

| Option | Default | Meaning |
|---|---|---|
| `model` | `nemo-fastconformer-en-80ms` | catalog name, archive name, local directory or archive URL |
| `language` | `None` | reported in transcripts; passed to SenseVoice/Whisper and as the Nemotron 3.5 prompt |
| `kind`, `files`, `sha256` | — | for models outside the catalog (see above) |
| `num_threads` | `2` | ONNX Runtime threads |
| `execution_provider` | `"cpu"` | `"cuda"` / `"coreml"` need a sherpa-onnx build with that provider (PyPI wheels are CPU) |
| `decoding_method` | `"greedy_search"` | or `"modified_beam_search"` (transducers; `max_active_paths`) |
| `endpoint_detection` | `False` | sherpa's endpoint rules (streaming models) |
| `tail_padding` | auto | silence (s) appended on `flush()`; auto = just enough for the last chunk |
| `interim_results` | `True` | emit `INTERIM_TRANSCRIPT` events |
| `word_timestamps` | `True` | word timings from token timestamps (models that report them) |
| `text_case` | `"auto"` | `"auto"` lowercases all-caps output, `"lower"`, `"keep"` |
| `max_segment_duration` | per model | offline models: split longer audio at pauses |
| `recognizer_options` | `{}` | extra keyword arguments for the sherpa-onnx factory (`hotwords_file`, `blank_penalty`…) |

### `SherpaOnnxTTS`

| Option | Default | Meaning |
|---|---|---|
| `model` | `piper-en_US-libritts_r-medium` | catalog name, archive name, local directory or archive URL |
| `voice` | model default | speaker id (`"12"`) or a Kokoro voice name (`"bm_george"`) |
| `speed` | `1.0` | speaking rate (0.25–4) |
| `sample_rate` | model rate | output rate; other rates are resampled |
| `lang` | `None` | Kokoro language hint (`"es"`, `"fr"`…) |
| `num_threads` | `2` | ONNX Runtime threads |
| `max_num_sentences` | `1` | sentences per synthesis step; each step's audio is emitted as soon as it is ready |
| `silence_scale` | `0.2` | pause between sentences of one request |
| `chunk_duration` | `0.05` | emitted chunk length (s) |

Audio streams out as sherpa generates it (per sentence); `stream()` uses the base
`SentenceStreamAdapter`. Cancelling a synthesis stops sherpa at the next sentence.

### `SherpaOnnxVAD`

`model` (`silero` / `ten-vad` / a local ONNX file with `kind=`), `num_threads` (1), and
`VADOptions` fields (`min_silence_duration=0.3`…) as keyword arguments.

## Latency

T1 benchmark (`van bench latency`, 13 turns × 2 sessions per condition, first turn of each
session excluded, run back to back) on an AMD Ryzen 5 5600 CPU (no GPU), with only the
recognizer changing: Silero VAD · Smart Turn v3.2 · Ollama `LiquidAI/lfm2.5-1.2b-instruct`
· Kokoro v1.0 fp16 (kokoro-onnx) · `cascade.first_sentence_max_chars: 40`, scenario
`benchmarks/scenarios/latency-local.yaml`.

| STT | v2v p50 | v2v p90 | end-of-turn delay p50 | STT flush → final p50 | dead air (> 2 s) |
|---|---:|---:|---:|---:|---:|
| faster-whisper `base` (CPU int8, batch) | 1,284 ms | 1,878 ms | 633 ms | 373 ms | 4 % |
| sherpa-onnx `nemo-fastconformer-en-80ms` | 1,209 ms | 1,872 ms | **401 ms** | **94 ms** | 13 % |
| sherpa-onnx `zipformer-en-kroko` | **1,126 ms** | **1,620 ms** | **400 ms** | 100 ms | 4 % |

* The streaming recognizer takes the final transcript off the critical path: the
  end-of-turn delay drops to the cascade's minimum endpointing delay (0.4 s with a turn
  detector); the STT final arrives in ≈ 100 ms, well inside that window.
* The rest of v2v is the TTS's first audio, which depends on the LLM's first clause and
  varies between runs (the lowercase, unpunctuated NeMo transcript changes the LLM's
  wording); Kroko's cased, punctuated output gave replies closest to the baseline.
* Transcripts of all 24 measured turns were correct with both streaming models.

With `latency-local-sherpa.yaml` (stimuli rendered by sherpa-onnx Kokoro int8) and the NeMo
model, 6 measured turns gave v2v p50 952 ms / p90 1,116 ms, end-of-turn delay 401 ms.

## Tests

```bash
uv run pytest -q tests/providers/test_sherpa_onnx.py            # fake sherpa module, offline
uv run pytest -q -m model tests/providers/test_sherpa_onnx.py   # real models, ≈ 110 MB download
```

The model tests use the smallest models of each family (`zipformer-en-kroko` 57 MB,
`moonshine-tiny-en` 30 MB, `piper-en_US-libritts_r-medium` 23 MB, `silero` VAD).
