# Moonshine Streaming (local streaming STT)

[Moonshine](https://github.com/moonshine-ai/moonshine) Streaming models are small
speech recognizers (34M–245M parameters, MIT licence for English) built for CPUs and edge
devices. voice-agent-next runs them through the native runtime of the
[`moonshine-voice`](https://pypi.org/project/moonshine-voice/) package (C++ on ONNX
Runtime, no torch). The runtime encodes audio incrementally while the user speaks, so the
final transcript is ready very soon after speech ends. Batch Whisper instead has to
transcribe the whole utterance at that point.

| | |
|---|---|
| Spec | `moonshine/<model>`, default model `small-streaming` |
| Class | `voice_agent_next.providers.moonshine.MoonshineSTT` |
| Extra | `pip install 'voice-agent-next[moonshine]'` (or `uv sync --extra moonshine`) |
| Credentials | none |
| Capabilities | streaming, interim results, word timestamps (opt-in); no language detection, no semantic end of turn |
| Platforms | Linux x86_64 (glibc ≥ 2.34) and aarch64, macOS 15+ on Apple Silicon, Windows x64. Every Python ≥ 3.11 (the wheels are `py3-none`) |

On other platforms (Intel Macs, 32-bit ARM, musl) the extra installs nothing and
constructing the provider raises `MissingDependencyError`.

## Usage

```python
from voice_agent_next import create

stt = create("stt", "moonshine/small-streaming")
await stt.warmup()  # download (first run) + load + warm-up pass

stream = stt.stream()
stream.push_audio(frame)  # any sample rate / channel count
stream.flush()  # external endpointing: finalize the current line now
async for ev in stream:  # START_OF_SPEECH, INTERIM_TRANSCRIPT, FINAL_TRANSCRIPT...
    print(ev.type, ev.text)
```

```yaml
# agent.yaml: a fully local CPU cascade
stt: {provider: moonshine/small-streaming, language: en}
vad: silero
turn_detector: smart_turn
llm: {provider: ollama/LiquidAI/lfm2.5-1.2b-instruct, temperature: 0}
tts: kokoro/v1.0-fp16
```

`MoonshineSTT` streams natively, so the cascade uses it directly and does not wrap it in a
`StreamAdapter`.

### Events

* The runtime has its own VAD that splits speech into *lines*. A new line emits
  `START_OF_SPEECH`, and each pass that changes its text emits `INTERIM_TRANSCRIPT`. When
  the runtime completes a line on its own, you get `FINAL_TRANSCRIPT` followed by
  `END_OF_SPEECH`. A pipeline without a VAD can use these for endpointing.
* In the cascade, the VAD and the turn detector own endpointing. At VAD end-of-speech the
  cascade calls `flush()`. The stream then stops the native stream (the runtime completes
  every line from the audio it has), emits one `FINAL_TRANSCRIPT` per completed line, and
  restarts the stream. If there was nothing left to complete, `flush()` emits an **empty**
  `FINAL_TRANSCRIPT`, so a caller waiting for "the final after my flush" never waits for a
  timeout. `flush()` never emits `END_OF_SPEECH`, because the caller already knows that
  speech ended.
* Transcript and word times are on the stream's clock: seconds of audio pushed since the
  stream opened, carried over across flushes.

### Threads

Every native call (adding audio, transcription passes, stop/start) runs in
`asyncio.to_thread`. While a pass runs, incoming audio is queued. The next call hands all
of it over at once, so a slow machine makes fewer, larger passes instead of falling
behind. One `MoonshineSTT` loads its model once, lazily and under a lock. All of its
streams share that model, and the runtime serializes their passes. For concurrent
sessions that must not wait for each other, create one instance per session.

## Models

The models come from the package's built-in catalog. The URL, size and CRC32C checksum of
every file are compiled into the native library, so a given `moonshine-voice` release
always downloads the same verified files. They are cached in the package's own cache:
`~/.cache/moonshine_voice` on Linux, the platform user-cache directory elsewhere, or
`$MOONSHINE_VOICE_CACHE`, or `cache_dir=`. With `VAN_OFFLINE=1` or `local_files_only=True`
nothing is downloaded, and a missing file raises `ProviderConnectionError`.

| Model | Params | Download | English WER (model card) | Notes |
|---|---|---|---|---|
| `tiny-streaming` | 34M | 45 MB | 12.0 % | fastest; Raspberry-Pi class |
| `small-streaming` — **default** | 123M | 142 MB | 7.8 % | best latency/accuracy trade-off on a desktop CPU |
| `medium-streaming` | 245M | 269 MB | 6.7 % | most accurate; ~2× the CPU of `small` |
| `tiny`, `base` | 27M / 61M | 44 / 141 MB | — | original non-streaming Moonshine (each pass re-transcribes the line) |

A local directory with converted models also works: `MoonshineSTT(model="/models/ms",
arch="small-streaming")`.

### Languages

`language` picks a model from the catalog (`"en"` by default; `"es-ES"` becomes `"es"`).
`moonshine-voice` 0.1.5 has these:

| Language | Models |
|---|---|
| English `en` | `medium-streaming`, `small-streaming`, `tiny-streaming`, `base`, `tiny` |
| Spanish `es` | `small-streaming`, `tiny-streaming`, `base` |
| German `de` | `small-streaming`, `tiny-streaming` |
| Japanese `ja` | `small-streaming`, `tiny-streaming`, `base`, `tiny` |
| Arabic `ar`, Vietnamese `vi`, Chinese `zh` | `tiny-streaming`, `base` |
| Tagalog `tl` | `tiny-streaming` |
| Korean `ko` | `tiny` |
| Ukrainian `uk` | `base` |

Asking for a model that does not exist for a language (`moonshine/medium-streaming` with
`language="de"`) raises `ConfigurationError`. **Licence:** only the English models are MIT.
The other languages are released under the non-commercial
[Moonshine Community License](https://www.moonshine.ai/license), and the provider logs a
warning when it loads one. There is no language detection: one instance serves one
language, and `stream(language=...)` with a different language raises `ConfigurationError`.

## Options

| Option | Default | Meaning |
|---|---|---|
| `model` | `small-streaming` | catalog model name, or a local model directory (with `arch=`) |
| `language` | `None` (English) | language code; see above |
| `arch` | `None` | architecture of a local model directory |
| `update_interval` | `0.5` | seconds of new audio between incremental passes (also the runtime's `transcription_interval`). `0.2` makes flushes ~60 ms faster and costs ~2.5× the CPU |
| `interim_results` | `True` | `False` only encodes while the user speaks and decodes when the line completes (`decode_incomplete_lines=false`); it uses less CPU and emits no interims |
| `word_timestamps` | `False` | fill `Transcript.words`; downloads an extra decoder (`decoder_with_attention.ort`) |
| `keyterms` | `None` | words or phrases to bias the decoder towards (names, jargon); streaming models only; no commas |
| `cache_dir` | `None` | model cache root |
| `local_files_only` | `False` | never download (also `VAN_OFFLINE=1`) |
| `options` | `{}` | raw runtime options (see [options.md](https://github.com/moonshine-ai/moonshine/blob/main/docs/api/options.md)), e.g. `{"vad_threshold": 0.6, "max_tokens_per_second": 13.0}`; they override the options above |

## Latency

Measured on an AMD Ryzen 5 5600 (6 cores / 12 threads, `powersave` governor), CPU only,
`moonshine-voice` 0.1.5. **Other agents' jobs were loading the same CPU during these runs**
(load average 1.5–3.5 on 12 threads). Treat the numbers as indicative.

**Recognizer alone** (JFK clip, 20 ms chunks, flushed 0.25 s after the last word):

| Model | pass every 0.5 s: CPU per audio second / flush | pass every 0.2 s: CPU per audio second / flush |
|---|---|---|
| `tiny-streaming` | 0.09 s / ~78 ms | 0.21 s / — |
| `small-streaming` | 0.17 s / ~140 ms | 0.42 s / ~78 ms |
| `medium-streaming` | 0.28 s / ~210 ms | — |

The flush time is mostly the decoder re-reading the last line. When the runtime's own VAD
has already completed the line (after ~0.4 s of silence), a flush returns in ~1 ms.

**T1 voice-to-voice** (`benchmarks/scenarios/latency-local-moonshine.yaml`, the six
`latency-local` questions voiced by Kokoro, 12 turns × 2 sessions, first turn of each
session excluded; Silero VAD, Smart Turn v3.2, Ollama `LiquidAI/lfm2.5-1.2b-instruct`,
Kokoro v1.0 fp16; the runs were made back to back):

MOONSHINE_T1_TABLE

End-of-turn delay is measured from the end of the user's speech until the turn is
committed to the LLM. With a turn detector the cascade waits at least 400 ms
(`min_endpointing_delay`) after speech ends. The STT only adds latency when its final
transcript takes longer than that, and a Moonshine flush almost never does. The remaining
v2v time is spent in the LLM and in Kokoro's first clause. Those depend on the reply text,
which differs slightly between recognizers (for example "Ask not!" vs "Ask not.").

To reproduce, follow the header of the scenario file.

## Errors

| Situation | Exception |
|---|---|
| `moonshine-voice` not installed / no wheel for the platform | `MissingDependencyError` (at construction, with the install command) |
| unknown model, no model for the language, bad option, keyterm with a comma | `ConfigurationError` |
| network failure while downloading, or model not cached while offline | `ProviderConnectionError` |
| native runtime failure (model load, transcription pass) | `ProviderError`, raised from the stream's iterator |

A failed load is retried on the next call.

## Limitations

* No language detection or semantic end-of-turn detection. Use a turn detector
  (`smart_turn`) in the cascade.
* A flush restarts the native stream. If the cascade flushes in the middle of a word (a
  very short VAD pause), the next line starts without that acoustic context.
* The runtime's VAD can end a line during a short pause. The cascade then receives two
  finals for one turn and joins them, which is harmless.
* Non-English models are non-commercial (Moonshine Community License).
* sherpa-onnx (#9) can also run the *offline* Moonshine models. This provider is the
  native streaming runtime.
