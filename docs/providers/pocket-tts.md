# Kyutai Pocket TTS (`pocket-tts`)

[Pocket TTS](https://github.com/kyutai-labs/pocket-tts) is Kyutai's small text-to-speech
model: about 100M parameters, 24 kHz mono output, English, French, German, Portuguese,
Italian, Spanish and Dutch. It generates 80 ms audio frames one after the other and decodes
them while it generates the next ones, so a sentence starts playing about 100 ms after it
was sent, on a CPU. It can also clone a voice from a short audio prompt.
voice-agent-next runs it locally with the `pocket-tts` package on CPU-only PyTorch. It needs
no GPU and no API key, and works offline once the weights are cached.

| | |
|---|---|
| Spec | `pocket-tts`, `pocket-tts/<voice>`, `pocket-tts/<language>`, `pocket-tts/<language>/<voice>` (alias: `pocket`) |
| Class | `voice_agent_next.providers.pocket_tts.PocketTTS` |
| Extra | `voice-agent-next[pocket-tts]` (`pocket-tts>=3.3,<4` and CPU-only `torch`) |
| Platforms | Linux (x86-64, arm64), Windows, macOS on Apple silicon. Not on Intel Macs: PyTorch stopped publishing wheels for them after 2.2. |
| Python | 3.11 to 3.14 |
| Output | 24 kHz, mono, s16le, in 80 ms frames, emitted as soon as they are decoded |
| Streaming | audio out: frame by frame. Text in: none, so `stream()` uses the `SentenceStreamAdapter` |
| Word timings | estimated, per sentence (see [Word timings](#word-timings)) |

## Setup

```bash
uv sync --extra pocket-tts              # in a checkout: CPU torch from download.pytorch.org
uv add 'voice-agent-next[pocket-tts]'   # in a uv project (add the same [tool.uv.sources] torch entry)
```

The repository's `pyproject.toml` points `torch` to the PyTorch CPU wheel index on Linux and
Windows (`[tool.uv.sources]`). With that, the wheel is 125 MB on Windows and 196 MB on
Linux x86-64, instead of the multi-GB CUDA build. macOS wheels on PyPI are CPU/MPS builds
already. **pip** does not read that setting. Install CPU torch first, or pip will pull the
CUDA build on Linux:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install 'voice-agent-next[pocket-tts]'
```

The weights are downloaded on first use through `huggingface_hub`, into the Hugging Face
cache (`~/.cache/huggingface/hub`). The pocket-tts configs pin every file to a repository
revision. The default English model is about 220 MB with its tokenizer, and every language
is a separate model of the same size. A predefined voice is a 6–8 MB state file. To prepare
an offline machine, run `await tts.warmup()` once while online (with every voice you need).
After that, set `HF_HUB_OFFLINE=1`.

## Usage

```python
from voice_agent_next import create

tts = create("tts", "pocket-tts")                    # English, voice "alba"
tts = create("tts", "pocket-tts/marius")             # another predefined voice
tts = create("tts", "pocket-tts/french")             # French, its default voice "estelle"
tts = create("tts", "pocket-tts/german/juergen", temperature=0.3)
await tts.warmup()  # download + load the model and the default voice now

audio = await tts.synthesize("Hello! How can I help you today?").collect()
async for chunk in tts.synthesize("Frames arrive while the sentence is still generated."):
    play(chunk.frame)  # AudioFrame, 24 kHz mono, 80 ms
```

In a config file:

```yaml
tts: {provider: pocket-tts/alba}
# or
tts: {provider: pocket-tts, language: spanish, voice: /path/to/me.wav}
```

### Options

| Option | Default | Meaning |
|---|---|---|
| `voice` | from the spec, else `alba` (English) or the language's default voice | predefined voice name, audio file to clone, exported `.safetensors` voice state, or an `hf://` / `https://` URL of either |
| `language` | `english` | `english`, `french`, `german`, `portuguese`, `italian`, `spanish`, `dutch` or ISO codes (`fr`...). Also the 24-layer variants (`french_24l`...) and dated English releases (`english_2026-04`) |
| `config` | – | custom pocket-tts YAML (path, URL, `hf://`), instead of `language`. Predefined voices do not work with custom weights |
| `temperature` | model default (0.3) | sampling temperature |
| `sampler_decode_steps` | 1 | flow decoding steps |
| `eos_threshold` | -4.0 | end-of-speech threshold (higher: the model speaks longer) |
| `frames_after_eos` | automatic | 80 ms frames generated after the end of speech |
| `quantize` | `False` | dynamic int8 quantization of the transformer (pocket-tts reports ~27 % faster on x86, ~48 % less RAM) |
| `num_threads` | pocket-tts default (1) | `torch.set_num_threads`. This setting is **process-wide**. pocket-tts already decodes on a second thread |
| `split_sentences` | `True` | synthesize long texts sentence by sentence (sooner first audio, per-sentence word timings) |
| `word_timings` | `True` | attach estimated word timings |
| `truncate_prompt` | `True` | use only the first 30 s of a voice prompt |
| `voice_cache_dir` | model cache `/pocket-tts/voices` | where cloned voice states are cached |

Inference runs on one dedicated worker thread per `PocketTTS` instance (the model is not
thread-safe), so requests are served one at a time in order. Closing a stream (barge-in)
stops the generation at the next frame.

## Voices

Every predefined voice is available for every language. Pick one whose accent fits the
language: the voice prompt carries the accent. The defaults are `alba` (English),
`estelle` (French), `juergen` (German), `rafael` (Portuguese), `giovanni` (Italian),
`lola` (Spanish) and `daan` (Dutch).

The voices come from [kyutai/tts-voices](https://huggingface.co/kyutai/tts-voices) and
other public datasets, with **different licenses**. Check before commercial use:

| Voices | Source | License |
|---|---|---|
| `alba` | alba-mackenna | CC-BY-4.0 |
| `marius`, `javert` | voice-donations | CC0 |
| `bill_boerst`, `peter_yearsley`, `stuart_bell`, `caro_davy` | voice-zero | CC0 |
| `anna`, `vera`, `fantine`, `charles`, `paul`, `eponine`, `azelma`, `george`, `mary`, `jane`, `michael`, `eve` | VCTK | CC-BY-4.0 |
| `cosette` | Expresso | **CC-BY-NC-4.0 (non-commercial)** |
| `jean` | EARS | **CC-BY-NC-4.0 (non-commercial)** |
| `estelle` | unmute-prod-website | see the repository README (mostly CC0, mixed) |
| `daan` | CML-TTS | CC-BY-4.0 |
| `giovanni`, `lola` | Common Voice | CC0 |
| `juergen`, `rafael` | kyutai/pocket-tts | not stated; check upstream |

The model weights are CC-BY-4.0 and the `pocket-tts` code is MIT.

## Voice cloning

Pass an audio file (WAV, MP3, FLAC..., any sample rate, ideally 5–30 s of clean speech)
as the voice:

```python
tts = create("tts", "pocket-tts", voice="me.wav")
await tts.load_voice("me.wav")        # optional: encode it now (a few seconds on CPU)
await tts.export_voice("me.wav", "me.safetensors")  # reusable state, loads in milliseconds
```

The encoded state is cached in memory (up to 8 voices) and on disk under
`voice_cache_dir`. The cache key is the file content plus the model, so the next process
skips the encoding. Changing the file invalidates its cache entry.

**Access.** Cloning from audio needs the `kyutai/pocket-tts` weights, which are **gated**
on Hugging Face. Accept the terms on <https://huggingface.co/kyutai/pocket-tts> and log in
(`hf auth login` or `HF_TOKEN`). Without access, pocket-tts silently uses the ungated
`kyutai/pocket-tts-without-voice-cloning` weights. Those play the predefined voices and
exported `.safetensors` states, but encoding a new audio prompt raises a
`ConfigurationError` that explains this. An exported state made on a machine with access
works on machines without it.

Only clone voices you have the right to use.

## Word timings

Pocket TTS has no text alignment. After each sentence is synthesized, the provider finds
the speech span of its audio (samples above 10 % of the peak). It then spreads the words
over that span in proportion to their letters, with pauses after commas and full stops.
The timings arrive right after the sentence's audio, as an item without audio. The
cascade uses them for word-exact truncation on barge-in. Within one sentence the error is
typically a word or two, and there is one blind spot: words of a sentence whose audio is
still being generated are not known yet. Generation runs 3–4× faster than real time, so
that window is short. `word_timings=False` turns this off (the cascade then truncates per
sentence).

## Latency

T1 benchmark on the local CPU cascade from [the results](../benchmarks/results.md): AMD
Ryzen 5 5600, `powersave` governor. Stack: Silero VAD, Smart Turn v3.2, faster-whisper
`base` int8, Ollama `lfm2.5-1.2b-instruct`. Scenario `latency-local`, 12 turns, the first
one reported as cold start. The runs were back to back on 2026-09-24 with other work on
the machine (load average ≈ 3):

| TTS | run | v2v p50 | v2v p90 | TTS first audio p50 | end-of-turn p50 | dead air |
|---|---|---:|---:|---:|---:|---:|
| Kokoro v1.0 fp16 | 1 | 1,429 ms | 2,824 ms | 536 ms | 754 ms | 36 % |
| **Pocket TTS** (alba) | 1 | **880 ms** | **991 ms** | **95 ms** | 623 ms | 0 % |
| Kokoro v1.0 fp16 | 2 | 1,279 ms | 1,829 ms | 442 ms | 619 ms | 0 % |
| **Pocket TTS** (alba) | 2 | **891 ms** | **1,184 ms** | **101 ms** | 624 ms | 0 % |

Kokoro renders a whole clause before emitting any audio (RTF ≈ 0.16, so 400 ms or more
per clause). Pocket TTS emits its first 80 ms frame after about 100 ms, whatever the
sentence length. This takes about 0.4–0.5 s off voice-to-voice latency and removes the
long-first-clause tail. Standalone, on the same machine: first audio 70–150 ms after
warm-up, with a real-time factor of about 0.27 (default single torch thread). With
`num_threads=2`, first audio is 70–85 ms.

Agent config used (swap the `tts` line for `{provider: kokoro/v1.0-fp16}`):

```yaml
stt: {provider: faster-whisper/base, device: cpu, compute_type: int8, language: en}
llm: {provider: ollama/LiquidAI/lfm2.5-1.2b-instruct, temperature: 0}
tts: {provider: pocket-tts/alba}
vad: silero
turn_detector: smart_turn
agent:
  instructions: >-
    You are a helpful customer-service voice assistant. Answer every question in two or
    three full sentences of plain spoken English, without lists or markdown.
```

```bash
van bench latency -c agent.yaml -s benchmarks/scenarios/latency-local.yaml --turns 12
```

`benchmarks/scenarios/latency-local-pocket.yaml` has the same questions, with the caller
voiced by Pocket TTS (`george`), for machines without Kokoro. Pocket TTS samples with a
temperature, so these stimuli differ slightly from run to run. The manifest records their
hashes.

## ONNX and sherpa-onnx ports (no torch)

Community ports run Pocket TTS on ONNX Runtime without PyTorch, notably
[sherpa-onnx](https://k2-fsa.github.io/sherpa/onnx/tts/all/) (`OfflineTts` with the
[`sherpa-onnx-pocket-tts-int8-2026-01-26`](https://huggingface.co/csukuangfj2/sherpa-onnx-pocket-tts-int8-2026-01-26)
model, made with KevinAHM's ONNX export tools).
They are not wrapped here, for these reasons:

* **Model version.** The sherpa-onnx model is an int8 English export dated 2026-01-26.
  The current English model (2026-09) and the other languages are torch-only.
* **Quality.** Users report that the output differs from the reference implementation
  (sherpa-onnx issue #3180, open).
* **License.** The sherpa-onnx model card marks the export as non-commercial, although the
  original weights are CC-BY-4.0.
* **Streaming.** sherpa-onnx exposes Pocket TTS through its `OfflineTts` API. We did not
  verify that it delivers audio frame by frame, which is the reason to use Pocket TTS.
  Otherwise that would need a custom ONNX Runtime loop over the exported graphs.

The trade-off is torch: about 125–200 MB to download and 730 MB installed, against about
40 MB for onnxruntime. Where torch is not acceptable, an ONNX backend is a follow-up.
