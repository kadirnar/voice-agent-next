# MLX on Apple silicon (`mlx`, `mlx_whisper`, `mlx_audio`, `mlx_lm`)

[MLX](https://github.com/ml-explore/mlx) is Apple's array framework for the GPU of Apple
silicon Macs (Metal, unified memory). Four providers run a whole cascade on it:

| Spec | Component | Runtime | Class |
|---|---|---|---|
| `mlx/parakeet-tdt-0.6b-v3` (default; alias `parakeet-mlx`) | STT, streaming | [parakeet-mlx](https://github.com/senstella/parakeet-mlx) | `providers.mlx.ParakeetMLXSTT` |
| `mlx_whisper/large-v3-turbo` (default) | STT, batch | [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) | `providers.mlx_whisper.MLXWhisperSTT` |
| `mlx_audio/kokoro` (default), `mlx_audio/pocket-tts` | TTS | [mlx-audio](https://github.com/Blaizzy/mlx-audio) | `providers.mlx_audio.MLXAudioTTS` |
| `mlx_lm`, `mlx_lm/<model>` | LLM | [mlx-lm](https://github.com/ml-explore/mlx-lm)'s `mlx_lm.server` | `providers.mlx_lm.MLXLMServerLLM` |

The `apple` preset uses them: `van run --preset apple` (see [Presets](../presets.md)).

```python
from voice_agent_next import AgentSession

session = AgentSession(
    stt="mlx/parakeet-tdt-0.6b-v3",
    llm="mlx_lm/mlx-community/Qwen3.5-4B-4bit",
    tts="mlx_audio/kokoro",
    vad="silero",
    turn_detector="smart_turn",
)
```

## Setup

MLX needs macOS on Apple silicon (M1 or later) and an arm64 Python. `van doctor` says
whether Python runs under Rosetta 2, where MLX cannot work.

```bash
pip install 'voice-agent-next[mlx]'           # Parakeet, mlx-audio, mlx-lm (+ openai client)
pip install 'voice-agent-next[mlx-whisper]'   # Whisper (pulls in torch, for its converter)
# in a checkout: uv sync --extra mlx --extra mlx-whisper
```

The extras carry `sys_platform == 'darwin' and platform_machine == 'arm64'` markers: on
Linux, Windows and Intel Macs they install nothing (only the `openai` client of `mlx`), and
the providers report "MLX runs on Apple silicon Macs only". Every provider module imports
without MLX, so `van providers` lists them everywhere.

Models are MLX conversions on the Hugging Face Hub, downloaded into the Hugging Face cache
on first use (or ahead of time with `van models download mlx/parakeet-tdt-0.6b-v3`, or with
`await component.warmup()`). `VAN_OFFLINE=1` forbids downloads.

**One MLX thread.** MLX keeps its GPU stream per thread and its arrays are not meant to be
shared across threads; the recognizer and the synthesizer also share one GPU. All MLX
providers of a process therefore run on one worker thread, in small steps: one audio chunk
of recognition, one segment (or streamed chunk) of synthesis. Recognition of a barge-in
waits for at most one synthesis step. The LLM runs in the `mlx_lm.server` process, so token
generation never occupies that thread.

## STT: Parakeet (`mlx`)

NVIDIA's Parakeet TDT 0.6B v3 transcribes 25 European languages with punctuation and casing
(v2 and the other Parakeets are English-only).

| Model | Download | Languages |
|---|---|---|
| `parakeet-tdt-0.6b-v3` | 2.5 GB | 25 European languages |
| `parakeet-tdt-0.6b-v2` | 2.5 GB | English |
| `parakeet-tdt_ctc-110m` | 459 MB | English |
| `parakeet-tdt-1.1b`, `parakeet-tdt_ctc-1.1b` | 4.3 GB | English |
| `parakeet-rnnt-0.6b`, `parakeet-ctc-0.6b` | 2.5 GB | English |

The downloads are float32; `dtype="bfloat16"` (the default) halves them in memory. Any
other parakeet-mlx checkpoint loads from its Hugging Face id (`mlx/<org>/<repo>`) or a
local directory with `config.json` and `model.safetensors`.

How a turn is recognized (the default, re-decoding streaming):

* while the user speaks, every `chunk_duration` (0.32 s) of new audio triggers an interim
  step: the utterance so far is decoded with full attention and emitted as an
  `INTERIM_TRANSCRIPT`. A step that takes longer than half of `chunk_duration` spaces the
  next one out, so the MLX thread stays at most about half busy;
* pending audio that is only silence (below about -50 dBFS) triggers no step: once the
  user stops talking the GPU goes idle, while the VAD and the turn detector wait for the
  end of the turn;
* `flush()` (the cascade calls it at the end of the turn) decodes the utterance once more
  and that is the `FINAL_TRANSCRIPT`: one full pass, identical to batch recognition. An
  utterance of only silence (a VAD false alarm) is not decoded;
* the next audio starts a new utterance (word times stay relative to the stream start).

Each step costs one pass over the utterance so far, which is what keeps the final
transcript fast and accurate for conversational turns (a few seconds to a few tens of
seconds). For long dictation, `incremental=True` switches to parakeet-mlx's cache-aware
`StreamingParakeet` (local attention with a rotating key/value cache, `context_size` and
`depth`): the cost per step stays bounded, but on the M1 runner its final transcript came
later (270 ms instead of 127 ms for a 3-second turn) and its accuracy depends on a large
right context (`(256, 256)` frames; with `(256, 32)` the JFK clip came out garbled).

Options: `streaming` (`False`: a batch recognizer that the cascade wraps in a
`StreamAdapter`), `interim_results`, `chunk_duration`, `incremental`, `context_size`,
`depth`, `dtype`, `beam_size` (1 = greedy), `language` (reported on transcripts; Parakeet v3
detects the language itself), `local_files_only`. Words carry timings and confidences
(sub-word tokens merged).

## STT: Whisper (`mlx_whisper`)

Whisper covers 99 languages. It is a batch recognizer: the cascade cuts the input into
utterances with the VAD and transcribes each one (a `StreamAdapter`), like
[faster-whisper](faster_whisper.md). Models: `tiny`, `tiny.en`, `base`, `base.en`, `small`,
`small.en`, `medium`, `medium.en`, `large-v3`, `large-v3-turbo` (default, 1.6 GB; alias
`turbo`), `distil-large-v3`, or an MLX Whisper repository id or directory. Decoding is
greedy (`temperature=0`) without conditioning on previous text; `fp16`,
`word_timestamps`, `initial_prompt` and `transcribe_options` (any `mlx_whisper.transcribe()`
keyword) are options. Several Whisper instances in one process keep their own model (no
reload between them).

## TTS: mlx-audio (`mlx_audio`)

| Model | Repository | Download | Voices | Streaming |
|---|---|---|---|---|
| `kokoro` | `mlx-community/Kokoro-82M-bf16` | 355 MB | 54 (`af_heart` default) | one chunk per text segment; the cascade sends sentences |
| `pocket-tts` | `mlx-community/pocket-tts` | 240 MB | `alba` (default), `marius`, `javert`, `jean`, `fantine`, `cosette`, `eponine`, `azelma`, or a WAV file to clone | audio chunks of `streaming_interval` seconds while the sentence is generated |

Both output 24 kHz mono. Other mlx-audio TTS models load by repository id
(`mlx_audio/mlx-community/<model>`); pass `sample_rate=` for their output rate if it is not
24 kHz (mlx-audio's output is resampled to the declared rate otherwise).

Options: `voice`, `speed`, `lang_code` (Kokoro; default from the voice's first letter:
`a` American and `b` British English, `e` Spanish, `f` French, `h` Hindi, `i` Italian,
`j` Japanese, `p` Portuguese, `z` Mandarin), `streaming_interval` (Pocket TTS, default
0.4 s), `generate_options` (extra keywords for the model's `generate()`),
`local_files_only`.

**Kokoro's G2P.** mlx-audio's Kokoro converts text to phonemes with
[misaki](https://github.com/hexgrad/misaki), which supports Python < 3.13 (the `mlx` extra
installs it there, without misaki's transformer backend that would pull in torch). English
also needs spaCy's `en_core_web_sm`; misaki installs it with pip on first use. In an
environment without pip (a uv venv), install it once:

```bash
uv pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
```

The provider checks both before loading and raises `MissingDependencyError` with that
command. Pocket TTS needs neither.

## LLM: mlx-lm server (`mlx_lm`)

```bash
python -m mlx_lm.server --model mlx-community/Qwen3.5-4B-4bit   # port 8080
```

`llm="mlx_lm"` asks for the model the server was started with (`default_model`);
`llm="mlx_lm/mlx-community/Qwen3.5-9B-4bit"` makes the server load (and download) that model
on first use, which `warmup()` does ahead of the first turn. The server address is
`base_url=`, else `MLX_LM_BASE_URL`, else `http://127.0.0.1:8080/v1`. It is an
[OpenAI-compatible host](openai-compatible.md): streaming, tool calls (mlx-lm parses the
tool-call formats of Qwen3/3.5, Gemma 4, Mistral, GLM, Kimi...), usage and every
`OpenAILLM` option work. Thinking is off by default
(`chat_template_kwargs={"enable_thinking": false}`), because reasoning delays the first
spoken word; `extra={"chat_template_kwargs": None}` restores the template's default.

**Why a server, not in-process generation.** Generating tokens keeps the GPU busy for
seconds. In-process, it would share the MLX thread (and the GIL) with recognition and
synthesis in the middle of a turn; in its own process it does not, one loaded model serves
every session, and mlx-lm already implements tool-call parsing and prompt caching. The
cost is one more process to start. The `apple` preset lists Ollama as the failover, so it
still runs when no mlx-lm server is up; `van presets` prints the command that starts it.
mlx-lm calls its server "not recommended for production" (it is a single-process HTTP
server); for several concurrent agents, LM Studio's MLX engine or Ollama are alternatives.

## Latency

Measured by `tests/providers/test_mlx_models.py` on a GitHub Actions `macos-latest` runner
(Apple M1, 3 cores, 7 GB; shared and virtualized, so slower than a desktop Mac), Python
3.12, mlx 0.32, 2026-09-25. Warm models; best of three where noted. Streaming tests push
the 11-second [JFK clip](https://github.com/openai/whisper/raw/main/tests/jfk.flac) (or its
first 3 seconds) in 20 ms frames at real-time speed, then flush.

| Component | Measurement | Result |
|---|---|---|
| `mlx/parakeet-tdt_ctc-110m` | final transcript after the flush, 3 s turn (streaming) | **127 ms** (5 interim results) |
| | same, 11 s utterance | 300 ms (18 interim results), text exact |
| | batch, 3 s turn | 104 ms |
| | `incremental=True`: 3 s turn / 11 s utterance | 270 ms / 488 ms |
| `mlx_whisper/tiny` | batch, 3 s turn (the final behind a `StreamAdapter`) | **97 ms** |
| | batch, 11 s clip | 1.3 s |
| `mlx_audio/pocket-tts` | first audio of a 70-character sentence | **140 ms** (176 ms median), RTF 0.37, 24 chunks |
| `mlx_audio/kokoro` | first audio of "Hello! How can I help you today?" (one segment) | 542 ms, RTF 0.23 |
| `mlx_lm` + `mlx-community/Qwen3-1.7B-4bit` | time to first token (short prompt) | **211 ms** (266 ms median) |
| | tool call | `get_weather({"city": "Paris"})` |

Loading (with the warm-up inference, models cached): Parakeet 110M 1.7 s, Whisper tiny
2.8 s, Pocket TTS 6.3 s, Kokoro 12 s. The default Parakeet (0.6B v3, 2.5 GB download) and
Qwen3.5-4B were not run on the runner (the CI budget keeps downloads small); expect a
larger model to cost more per pass, roughly in proportion to its size.

Kokoro synthesizes a whole segment before the first audio; the cascade sends it one
sentence at a time, so a short first sentence starts sooner. Pocket TTS streams audio while
it generates, which is why its first audio does not depend on the sentence's length.

## Limitations

* Apple silicon only; the unit tests run everywhere against fake MLX modules.
* Parakeet's `language` is not detected per utterance (v3 recognizes the language but
  does not report it).
* Kokoro through mlx-audio needs Python < 3.13 (misaki); Pocket TTS works on every
  supported Python.
* mlx-audio's Kokoro and Pocket TTS have no word alignment: `SynthesizedAudio.words` is
  empty.
