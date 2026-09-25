# NeMo-Speech.cpp (`stt` / `tts: nemo-speech-cpp`)

[NeMo-Speech.cpp](https://github.com/NVIDIA/NeMo-Speech.cpp) is NVIDIA's ggml runtime
for its speech models. It runs on CPU, CUDA, Metal and Vulkan, on Linux, macOS and
Windows. Its `nemo-speech serve` command loads one ASR model and/or one TTS model. It
serves them through an OpenAI-compatible HTTP API and a realtime transcription
WebSocket. This provider is a client for that server, and can also run the server for
you.

It is the local GPU streaming STT the research note recommends (§4.2, §9.2):
**Nemotron Speech Streaming** and **Nemotron 3.5** are cache-aware RNNT models. They
decode 80–1120 ms chunks as the audio arrives, so the final transcript is ready about
40 ms after the audio ends.

| | |
|---|---|
| Specs | `nemo-speech-cpp/nemotron-en` (`stt`, default), `nemo-speech-cpp/nemotron-3.5`, `nemo-speech-cpp/parakeet-tdt`, `nemo-speech-cpp/parakeet-ctc`; `nemo-speech-cpp/magpie` (`tts`) |
| Classes | `voice_agent_next.providers.nemo_speech_cpp.NeMoSpeechCppSTT`, `NeMoSpeechCppTTS`, `NeMoSpeechCppServer` |
| Extra | none: plain WebSocket and HTTP (`websockets`, `httpx`) |
| Server | `nemo-speech` 0.1.0 (native binary, not on PyPI) |
| Environment | `NEMO_SPEECH_BASE_URL` (default `http://127.0.0.1:8080/v1`), `NEMO_SPEECH_API_KEY` (only if the server was started with `--api-key`), `NEMO_SPEECH_BIN` (the binary, for the managed server) |

## Install the server

NVIDIA publishes installers and prebuilt archives
([install guide](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/main/docs/install.md)).
They install without `sudo`. The installer picks CUDA when `nvidia-smi` works, Metal on
Apple silicon, and CPU otherwise.

=== "Linux / macOS"

    ```bash
    curl -fsSL https://github.com/NVIDIA/NeMo-Speech.cpp/raw/main/scripts/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    nemo-speech doctor          # features, devices (e.g. "[0] gpu NVIDIA GeForce RTX 5070 Ti")
    # force a backend: ... | sh -s -- --backend cpu   (or vulkan)
    ```

=== "Windows (PowerShell)"

    ```powershell
    irm https://github.com/NVIDIA/NeMo-Speech.cpp/raw/main/scripts/install.ps1 | iex
    # new window, then:
    nemo-speech doctor
    ```

=== "Manual"

    Download `nemo-speech-<version>-<os>-<arch>-<backend>` (`cpu`, `cuda`, `vulkan`,
    `metal`) and its `.sha256` from the
    [releases](https://github.com/NVIDIA/NeMo-Speech.cpp/releases), verify it, extract it
    anywhere, and set `NEMO_SPEECH_BIN=/path/to/bin/nemo-speech`.

The Linux x86-64 CUDA archive bundles its CUDA user-space libraries and only needs an
NVIDIA driver. It supports compute capability 7.5 (RTX 20-series) and newer, including
Blackwell: it ran on an RTX 5070 Ti.

## Start the server

```bash
nemo-speech serve --asr-model nemotron-en                   # streaming STT on :8080
nemo-speech serve --asr-model nemotron-en --tts-model magpie --port 8080   # + Magpie TTS
nemo-speech serve --asr-model nemotron-en --asr.streaming.rnnt_right_context=0  # 80 ms chunks
```

An indexed model name (`nemotron-en`, `nemotron-3.5`, `parakeet-tdt`, `parakeet-ctc`,
`magpie`) is downloaded on first start into the server's cache and checked against a
pinned SHA-256. The cache is `~/.cache/nemo-speech/models` on Linux,
`~/Library/Caches/NeMoSpeech/models` on macOS and `%LOCALAPPDATA%\NeMoSpeech\models` on
Windows. `--device cpu|cuda[:N]|metal|vulkan[:N]` picks the backend (default `auto`).
The server binds to `127.0.0.1` by default. To expose it, set `--host 0.0.0.0` together
with `--api-key` (or `NEMO_SPEECH_HTTP_API_KEY`).

Or let the provider run it (`serve: true`, below). Use
`python -m voice_agent_next.providers.nemo_speech_cpp command --asr-model nemotron-en --chunk-ms 160`
to print the command line it would use.

## Usage

```python
from voice_agent_next import AgentSession

# a server you started
session = AgentSession(stt="nemo-speech-cpp/nemotron-en", llm=..., tts=..., vad="silero")

# a managed server: started on a free port on warmup(), stopped on aclose()
session = AgentSession(
    stt={
        "provider": "nemo-speech-cpp/nemotron-en",
        "serve": True,
        "server_options": {"chunk_ms": 160, "device": "cuda"},
    },
    llm=...,
    tts=...,
    vad="silero",
    turn_detector="smart_turn",
)
```

```yaml
# agent.yaml
stt: {provider: nemo-speech-cpp/nemotron-en, language: en-US}   # NEMO_SPEECH_BASE_URL or :8080
tts: {provider: nemo-speech-cpp/magpie, voice: John, language: en-US}
vad: silero
turn_detector: smart_turn
```

To run one server for both STT and TTS, share a `NeMoSpeechCppServer`:

```python
from voice_agent_next.providers.nemo_speech_cpp import (
    NeMoSpeechCppServer,
    NeMoSpeechCppSTT,
    NeMoSpeechCppTTS,
)

server = NeMoSpeechCppServer(asr_model="nemotron-en", tts_model="magpie", chunk_ms=160)
stt = NeMoSpeechCppSTT(server=server)  # the first request starts it
tts = NeMoSpeechCppTTS(server=server)
...
await server.stop()
```

The managed server looks for `nemo-speech` in `executable=`, then `NEMO_SPEECH_BIN`,
then `PATH`, then the installers' default locations. For catalog ASR models
(`van models list nemo-speech-cpp`), the provider downloads the pinned GGUF itself
(SHA-256 verified, into the Hugging Face cache) and passes the path to the server. Any
other name is handed to the server as is. The API key reaches the server through the
environment, never on the command line.

## Models

| Spec | Model | Streaming | Languages | Size (Q8) | License |
|---|---|---|---|---|---|
| `nemotron-en` (default) | [Nemotron Speech Streaming EN 0.6B](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b) | cache-aware, 80–1120 ms | English | 700 MB | NVIDIA Open Model License |
| `nemotron-3.5` | [Nemotron 3.5 ASR streaming 0.6B](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) | cache-aware, 80–1120 ms | 40+ locales (`language="auto"` detects) | 742 MB | NVIDIA OML (OpenMDW 1.1) |
| `parakeet-tdt` | [Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | **no** (batch only) | 25 European | 714 MB | CC-BY-4.0 |
| `parakeet-ctc` | [Parakeet CTC 1.1B](https://huggingface.co/nvidia/parakeet-ctc-1.1b) | buffered | English | 1.18 GB | CC-BY-4.0 |
| `magpie` (TTS) | [Magpie TTS Multilingual 357M](https://huggingface.co/nvidia/magpie_tts_multilingual_357m) + NanoCodec 22 kHz | no (whole reply per request) | en, es, de, fr, it, vi, hi (+ ja, zh in source builds) | 449 MB + codec | NVIDIA Open Model License |

The server serves the one ASR model it loaded, whatever the spec says. On an external
server, the model name only decides whether the provider streams: `parakeet-tdt` makes
it a batch STT, and the cascade then segments the audio with its VAD.

### Chunk size

The cache-aware models trade latency against accuracy through the chunk size, and the
chunk size is a **server** setting: `asr.streaming.rnnt_right_context` equals
chunk / 80 ms − 1. You set it with `NeMoSpeechCppServer(chunk_ms=...)` or the
`--asr.streaming.rnnt_right_context` flag. The models were trained for 80, 160
(server default), 560 and 1120 ms. The STT's own `chunk_ms` (default 20 ms) is only the
audio per WebSocket frame. The server answers every frame with one partial, so small
frames give the freshest partials.

## Options

`NeMoSpeechCppSTT(...)`:

| Option | Default | |
|---|---|---|
| `model` | `nemotron-en` | see *Models* (or a GGUF path, for the managed server) |
| `base_url` | `NEMO_SPEECH_BASE_URL` or `http://127.0.0.1:8080/v1` | |
| `api_key` | `NEMO_SPEECH_API_KEY` | sent as `Authorization: Bearer` |
| `language` | model default | `en-US`, `es-ES`...; `auto` for Nemotron 3.5 |
| `streaming` | `True` unless offline-only | `False`: batch STT (`/v1/audio/transcriptions`) |
| `word_timestamps` | `False` | word timings on finals, in seconds from the stream start |
| `automatic_punctuation`, `verbatim`, `profanity_filter` | `True`, `False`, `False` | server post-processing (`verbatim` skips ITN) |
| `speech_contexts`, `prompt` | – | word boosting: `[{"phrases": ["Kowalczyk"], "boost": 3}]`; `prompt` boosts one phrase |
| `endpointing_ms` | server default | end-of-utterance silence, when server endpointing is on |
| `chunk_ms` | `20` | audio per WebSocket frame |
| `serve`, `server`, `server_options` | – | managed server (see above) |

`NeMoSpeechCppServer(...)`: `asr_model`, `tts_model`, `device`, `chunk_ms`,
`endpointing` (server-side end of utterance, off by default), `endpointing_ms`, `host`,
`port`, `api_key`, `args` (any other `--asr.*` / `--tts.*` flag), `env`, `executable`,
`startup_timeout`.

`NeMoSpeechCppTTS(...)`: `voice` (a local name from `GET /v1/models`, such as `John`, a
model-qualified `magpietts.John`, or a speaker index; default: the server's speaker),
`language`, `sample_rate` (default 22050; other rates are resampled), plus the options
of [OpenAI TTS](openai.md). `speed` must be 1.0.

## How it works

**Streaming STT.** `STT.stream()` opens
`ws://host:port/v1/audio/transcriptions/realtime`. This protocol is specific to the
project, not the OpenAI Realtime API. The provider sends one `session.update`
(`sample_rate: 16000`, language, post-processing options), then binary little-endian
PCM16 frames. The server's events map to STT events like this:

| Server event | STT event |
|---|---|
| `conversation.item.input_audio_transcription.delta` (`delta` = new text suffix, usually `""`) | `INTERIM_TRANSCRIPT` with the accumulated text; `START_OF_SPEECH` before the first one |
| `conversation.item.input_audio_transcription.completed` (`transcript`, `words`) | `FINAL_TRANSCRIPT`, then `END_OF_SPEECH` |
| `input_audio_buffer.committed` | acknowledges a flush |
| `error` | the stream fails with `ProviderError` |

`STTStream.flush()` sends `input_audio_buffer.commit`. The server finishes the stream and
answers with the final transcript, or with an empty one when nothing was said, so every
flush gets a final. A commit also resets the model's recognition state: the next audio
starts a fresh stream. The provider keeps word timestamps on one clock for the whole
session. The server closes a socket with a wrong API key with code 1008, which becomes
`AuthenticationError`.

**Server endpointing** (`--asr.endpointing.enable`, `NeMoSpeechCppServer(endpointing=True)`)
emits finals mid-stream after `endpointing_ms` of trailing silence (default 800 ms). It
is off by default: in a cascade, the VAD and turn detector decide when the user stopped
and flush the stream. On the LibriSpeech smoke set, server endpointing split 23 of 50
utterances at short pauses, and NeMo-Speech.cpp 0.1.0 dropped a word at some of these
boundaries ("if ~~worse~~ comes to worst"). WER at 160 ms chunks rose from 2.6 % to 4.1 %.

**Batch STT.** `transcribe()` posts a WAV to `/v1/audio/transcriptions`, the
OpenAI-compatible multipart subset (`response_format=json`, or `verbose_json` for
words).

**TTS.** Magpie runs through `POST /v1/audio/speech` with
`{"input", "voice", "language", "response_format": "wav"}`. The server synthesizes the
whole text before it answers, so the cascade sends one sentence per request and prefetches
the next one while the current one plays.

## Measurements

Measured on an RTX 5070 Ti (16 GB) with a Ryzen 5 5600, NeMo-Speech.cpp 0.1.0
(`linux-x86_64-cuda`), `nemotron-speech-streaming-en-0.6b.q8_0.gguf`, and the
50-utterance LibriSpeech test-clean smoke subset (`van bench asr`). Streaming runs at
real time in 20 ms chunks. TTFS is the time from the end of the audio to the final
transcript.

| System | Mode | WER | CER | TTFS p50 | TTFS p90 | First partial p50 | RTFx |
|---|---|---:|---:|---:|---:|---:|---:|
| NeMo-Speech.cpp Nemotron EN, 80 ms chunks (GPU) | streaming | 2.64 % | 0.66 % | 41 ms | 50 ms | 921 ms | 1.0 |
| NeMo-Speech.cpp Nemotron EN, 160 ms chunks (GPU) | streaming | 2.64 % | 0.66 % | 41 ms | 45 ms | 1,000 ms | 1.0 |
| NeMo-Speech.cpp Nemotron EN, 560 ms chunks (GPU) | streaming | 2.11 % | 0.54 % | 41 ms | 44 ms | 1,161 ms | 1.0 |
| NeMo-Speech.cpp Nemotron EN, 1120 ms chunks (GPU) | streaming | 2.02 % | 0.53 % | 41 ms | 43 ms | 1,241 ms | 1.0 |
| NeMo-Speech.cpp Nemotron EN (GPU) | batch (HTTP) | 1.67 % | 0.44 % | 22 ms | 39 ms | – | 339 |
| sherpa-onnx `nemo-fastconformer-en-80ms` (CPU) | streaming | 2.29 % | 0.71 % | 64 ms | 96 ms | 948 ms | 1.0 |
| faster-whisper `small.en` (CUDA) | batch | 2.82 % | 0.71 % | 140 ms | 244 ms | – | 52 |

The final arrives about 41 ms after the flush at every chunk size: the commit decodes only
the audio still buffered. The chunk size trades the time to the first partial (counted
from the start of the audio, leading silence included) against accuracy. Larger chunks
see more right context. Batch recognition of the complete utterance over HTTP remains the
most accurate. sherpa-onnx and faster-whisper `small.en` ran with the same harness.
faster-whisper `large-v3-turbo` was not measured: its 1.6 GB download exceeds this
machine's budget.

**T1 voice-to-voice latency.** The same local cascade was run three times, changing only
the STT: LFM2.5-1.2B on Ollama, Kokoro v1.0 fp16, Silero VAD and Smart Turn, on
`benchmarks/scenarios/latency-local-gpu.yaml` (6 turns × 2 sessions, 10 measured).
Nemotron streamed at 160 ms chunks.

| STT | v2v p50 | v2v p90 | STT final latency p50 | end-of-turn delay | dead air |
|---|---:|---:|---:|---:|---:|
| NeMo-Speech.cpp Nemotron EN (GPU, streaming) | 1,163 ms | 1,730 ms | **60 ms** | 401 ms | 0 % |
| sherpa-onnx `nemo-fastconformer-en-80ms` (CPU, streaming) | 1,255 ms | 1,593 ms | 119 ms | 401 ms | 10 % |
| faster-whisper `small.en` (CUDA, batch) | 1,222 ms | 2,239 ms | 95 ms | 401 ms | 20 % |

With the cascade, the final transcript comes about 60 ms after the flush, half the time
of the other two. The rest of the voice-to-voice time is the turn detector's hold and
Kokoro's time to first audio on the CPU. Those dominate here and vary from turn to turn,
so the v2v confidence intervals overlap.

## Limitations

- One model per capability per server. To switch models, restart the server (or run
  several on different ports).
- The partials only append text: the server sends the new suffix, or the whole text when
  its hypothesis changed in a way that is not a pure extension. The protocol cannot tell
  these two cases apart, so a rewritten partial shows up appended. Finals are always
  exact.
- There is no reconnect. A server that goes away fails the stream with
  `ProviderConnectionError`, and the session's error handling or failover takes over.
- Magpie is not a streaming TTS here: the HTTP subset returns the whole utterance. It was
  implemented from the upstream API reference and source, and was not run on this
  machine.
- The full-duplex VoiceChat WebSocket (`/v1/realtime` with a speech-to-speech model) is not
  covered by this provider.
