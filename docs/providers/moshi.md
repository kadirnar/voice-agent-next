# Moshi and PersonaPlex (`moshi`, `personaplex` engines)

Local, open, **full-duplex** speech-to-speech. The engine connects to a running Moshi
server over its WebSocket protocol, and the model listens and speaks at the same time.
It works with Kyutai's `python -m moshi.server`, the Rust `moshi-backend` and
`moshi_mlx.local_web`, and with NVIDIA's PersonaPlex server. The code is in
`voice_agent_next/providers/moshi.py` and `providers/personaplex.py`.

```python
from voice_agent_next import Agent, AgentSession
from voice_agent_next.providers.moshi import MoshiEngine

session = AgentSession(MoshiEngine(base_url="ws://localhost:8998"))  # or AgentSession("moshi")
await session.run(Agent(""), transport)

session = AgentSession("personaplex")  # wss://localhost:8998
await session.run(Agent("You work for Acme Bank and your name is Sam.", voice="NATM1"), transport)
```

```yaml
# agent.yaml
engine: {provider: moshi/moshika-q8, url: "ws://localhost:8998"}
```

## Setup

```bash
pip install 'voice-agent-next[moshi]'     # sphn: Kyutai's Ogg/Opus codec (wheels, no libopus)
```

The client runs anywhere: Linux, macOS or Windows. The server needs a GPU, or Apple
Silicon for MLX. The model sits behind the server, so the `model` in the spec
(`moshi/moshika-q8`) is only a label for metrics.

### Running a server

| Platform | Command | Memory |
|---|---|---|
| Linux / NVIDIA 16 GB, PyTorch int8 | `pip install moshi` then `python -m moshi.server --hf-repo kyutai/moshika-pytorch-q8 --static none` (needs the workaround below) | 10 GB VRAM measured (10,154 MiB; 8.4 GB download) |
| Linux / NVIDIA 24 GB+, PyTorch bf16 | `python -m moshi.server --hf-repo kyutai/moshiko-pytorch-bf16` | about 24 GB |
| Linux / NVIDIA, Rust (candle) q8 | in `kyutai-labs/moshi/rust`: `cargo run --features cuda --bin moshi-backend -r -- --config moshi-backend/config-q8.json standalone` (needs `nvcc`), then use `url="wss://localhost:8998"` (self-signed certificate) | about 9 GB |
| macOS / Apple Silicon | `pip install moshi_mlx` then `python -m moshi_mlx.local_web -q 4` (`-q 8`, `--hf-repo kyutai/moshika-mlx-q4`) | 16 GB unified memory for q4 |
| Windows | Kyutai does not support Windows servers. Run the server under WSL2 or on another machine and point `url` at it | |

Pick Moshika (female voice) or Moshiko (male voice) with `--hf-repo`. `moshi.server`
serves **one conversation at a time**. A second client gets no handshake until the first
one leaves, and the engine then raises `ProviderTimeoutError` after `connect_timeout`.

**int8 with moshi 0.2.13.** When `moshi.server` loads a checkpoint, it casts every float
tensor to bf16. That includes the int8 scales (`*_scb`) that bitsandbytes needs in
float32, so the q8 checkpoints crash at warm-up with `Expected weight_scb to have type
float`. Until upstream fixes this, start the server through this wrapper:

```python
# moshi_q8_server.py — run as: python moshi_q8_server.py --hf-repo kyutai/moshika-pytorch-q8 ...
import runpy
from moshi.models import loaders

_load_file = loaders.load_file


class _KeepScales(dict):
    def __setitem__(self, key, value):
        if key.endswith("_scb") and key in self:
            return  # keep the float32 scale, not the down-cast copy
        super().__setitem__(key, value)


loaders.load_file = lambda *a, **kw: _KeepScales(_load_file(*a, **kw))
runpy.run_module("moshi.server", run_name="__main__")
```

On Blackwell GPUs (RTX 50xx) install a CUDA 12.8 PyTorch first, e.g.
`pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128`.

### PersonaPlex

PersonaPlex-7B (NVIDIA Open Model License) is a Moshi fine-tune that takes a **text role
prompt** and a **voice prompt**. It uses the same protocol, and its server
(`NVIDIA/personaplex`, `pip install moshi/.`, `SSL_DIR=$(mktemp -d); python -m
moshi.server --ssl "$SSL_DIR"`, add `--cpu-offload` on small GPUs) reads both prompts
from the query string:

* `text_prompt` comes from the agent's `instructions`. If they are empty, `text_prompt=`
  is used, which defaults to "You enjoy having a good conversation.". The PersonaPlex
  README has the prompting guide.
* `voice_prompt` comes from `Agent(voice=...)`, else `voice=`, else `NATF2`. The voices
  are `NATF0`–`NATF3`, `NATM0`–`NATM3`, `VARF0`–`VARF4` and `VARM0`–`VARM4`. A file
  name such as `custom.wav` is used as is.

The prompts are fixed when the server session starts, so `update(instructions=...)` only
takes effect after a reconnect. The server works through the prompts before it sends the
handshake, which takes a few seconds. The weights are 16.7 GB in bf16 and did not fit the
16 GB test GPU, so PersonaPlex was verified only against the fake server. The request
format was checked against the `NVIDIA/personaplex` server code.

## Protocol

`GET /api/chat` upgrades to a WebSocket that carries only binary messages. The first byte
of each message gives its kind:

| Byte | Kind | Direction | Payload |
|---|---|---|---|
| `0x00` | handshake | server → client | empty (`moshi.server`), or two LE `u32` values, protocol and model version (Rust) |
| `0x01` | audio | both | Ogg/Opus pages, 24 kHz mono. The first page of each stream carries `OpusHead`/`OpusTags` |
| `0x02` | text | server → client | one UTF-8 token of the inner monologue (SentencePiece `▁` replaced by a space) |
| `0x03` | control | — | start / endTurn / pause / restart (the servers ignore it) |
| `0x04` | metadata | server → client | JSON: sampling settings and model files (Rust backend, sent before the handshake) |
| `0x05` | error | server → client | UTF-8 text |
| `0x06` | ping | — | ignored |
| `0x07` | coloured text | server → client | colour byte + UTF-8 text |

The server runs one model step for every 80 ms (1920 samples) of **received** audio, so
the client's input clocks its output. If the client stops sending, the model stops
talking. The transports stream continuous audio, silence included. When a transport
stalls, the engine sends silence itself (`keepalive=True`).

The Rust server sends sampling query parameters (`text_temperature`, `text_topk`,
`audio_temperature`, `audio_topk`, `pad_mult`, `repetition_penalty(_context)`,
`text_seed`/`audio_seed`). Pass them as engine arguments or with `seed=`. `moshi.server`
ignores them.

## Full duplex and the event protocol

Moshi has no request/response cycle. It speaks when it wants, backchannels, yields when
it is talked over and never needs cancelling. The engine maps that stream onto the
turn-based [events](../../src/voice_agent_next/events.py) so that sessions, transcripts,
metrics and the benchmark work unchanged:

| Concept | Meaning for Moshi |
|---|---|
| **response** | One stretch of agent speech. It starts at the first text token or output chunk above `speech_threshold_db` (-40 dBFS; measured: speech -15…-25, silence -55…-80). It ends after `response_gap` (0.64 s) with neither, or after `yield_gap` (0.24 s) while the user talks. Audio between responses is not forwarded. `preroll` (80 ms) keeps the attack of the first syllable |
| `ResponseText` | the text tokens, the transcript of what Moshi says |
| `InputSpeechStarted/Stopped` | a local energy VAD on the audio sent to the server (`user_vad_threshold_db`, `user_min_silence`). They are reported **only while the agent is quiet**. If the user is still talking when Moshi yields, the start is reported `handover_delay` (0.3 s) later, once the session has played the tail of the response |
| `InputCommitted` | emitted before the first response that follows user speech: that response *is* the turn. If Moshi takes the floor while the user still talks, `InputSpeechStopped` comes first |
| `InputTranscript` | none: Moshi does not transcribe the user (`input_transcription=False`). The user's history entries stay empty |
| interrupt / `cancel_response()` | the model cannot be stopped. The response ends as `cancelled` and its audio is muted until Moshi's next pause |
| `create_response()` / `say()` / `send_text()` / tools | not supported (`text_input=False`, `tool_calling=False`). Calls are ignored with a warning, and so is `Agent(greeting=...)`: Moshi greets by itself |

**Why user speech over the agent goes unreported.** The session treats
`InputSpeechStarted` during a response as a barge-in: it pauses playback and, once the
interruption is confirmed, cancels. Moshi resolves overlaps itself: it yields,
backchannels or keeps talking, and the audio it produces already reflects that decision.
Pausing or muting it would only make the client disagree with the model. Keep the
default `SessionOptions` for this reason. `report_overlap=True` reports overlapping
speech anyway and hands the decision to the session's interruption policy, which mutes
Moshi (see above).

`capabilities`: `full_duplex=True`, `server_turn_detection=True`, `truncation=False`,
`tool_calling=False`, `text_input=False`, `input_transcription=False`,
`output_transcription=True`, and 24 kHz in and out.

**Echo.** A full-duplex model hears whatever the microphone picks up. Without echo
cancellation it hears itself and yields to its own voice. Use a headset, a browser or
WebRTC transport (both have built-in AEC), or the `aec` extra.

## Reconnects and metrics

If the connection drops (server restart, or the Rust backend's limit of 4500 steps = 6
minutes), the open response ends as `incomplete`. The engine then emits
`EngineStatus("reconnecting")` and retries with backoff (`max_reconnect_attempts`). When
it succeeds it emits `EngineStatus("reconnected")`. The new server session starts from
scratch, because Moshi cannot restore a conversation. Audio sent during the reconnect is
dropped: a real-time model cannot use stale audio. A rejected reconnect (HTTP 401/403) or
running out of attempts emits a non-recoverable `EngineErrorEvent`, and the connection
closes. `0x05` messages become recoverable `EngineErrorEvent`s and are kept in
`conn.errors`.

`EngineMetrics` per response:

* `ttfb` is the time from the local end of user speech to the first audio of the
  response. It is `None` for unprompted speech or when Moshi started while the user was
  talking.
* `output_audio_tokens` counts Mimi frames (12.5 Hz). `output_text_tokens` counts text
  tokens.
* `input_audio_tokens` counts the frames heard since the previous response.
* The server reports no usage: these audio counts are derived from durations, so
  `tokens_estimated` is `True`.

The connection also exposes `lag` and `max_lag` (input sent minus output received: how far
the server trails), `server_metadata`, `server_version` and `connections`.

## Options

| Argument | Default | Meaning |
|---|---|---|
| `url` | `ws://localhost:8998` | server origin (`http(s)://` also works). `/api/chat` is appended unless the URL has a path |
| `ssl_verify` | auto | verify `wss://` certificates. Off for `localhost` (self-signed), on elsewhere |
| `text_temperature`, `text_topk`, `audio_temperature`, `audio_topk`, `pad_mult`, `repetition_penalty`, `repetition_penalty_context`, `seed` | server default | sampling query parameters (Rust server) |
| `query` | `{}` | extra query parameters |
| `frame_duration` | 0.08 | Opus frame length sent (0.02/0.04/0.06/0.08 s). The server steps every 80 ms anyway |
| `speech_threshold_db` | -40 | output level that counts as agent speech |
| `response_gap` / `yield_gap` | 0.64 / 0.24 | agent silence that ends a response (while the user talks) |
| `preroll` | 0.08 | audio before a detected onset included in the response |
| `handover_delay` | 0.3 | delay before reporting user speech that continues after a response |
| `report_overlap` | `False` | report user speech over the agent (session barge-in policy) |
| `user_vad_threshold_db` / `user_min_silence` | -40 / 0.3 | local user VAD |
| `keepalive` | `True` | stream silence when the transport stalls |
| `connect_timeout` | 30 | WebSocket + handshake timeout |
| `reconnect` / `max_reconnect_attempts` | `True` / 5 | reconnect policy |

`PersonaPlexEngine` adds `voice=` and `text_prompt=`, and its `url` defaults to
`wss://localhost:8998`.

## Latency (measured)

T1 latency, [`benchmarks/scenarios/latency-local-moshi.yaml`](../../benchmarks/scenarios/latency-local-moshi.yaml):
Kokoro-spoken questions over the loopback transport, 3 sessions × 12 turns with the first
turn of each excluded. Moshika int8 (`kyutai/moshika-pytorch-q8`, moshi 0.2.13, torch
2.9.1+cu128) ran on an RTX 5070 Ti 16 GB (10,154 MiB VRAM) on the same machine.

```bash
van bench latency --engine '{provider: moshi/moshika-q8, url: "ws://127.0.0.1:8998"}' \
    -s benchmarks/scenarios/latency-local-moshi.yaml --turns 12 --sessions 3
```

| metric | n | p50 | p90 |
|---|---|---|---|
| voice-to-voice (recording), all turns | 33 | **276 ms** | **444 ms** |
| voice-to-voice, turns where Moshi waited for the end of the question | 22 | 344 ms | 446 ms |
| premature onsets (Moshi answered in a pause of the question) | 11 / 33 | | |
| missed / dead air | 0 % / 0 % | | |
| model lag (`max_lag`, input sent → output received) | | ~0.1 s | 0.3 s max |

**How full-duplex v2v is measured.** It is measured the same way as for every other
engine: on the call recording, from the annotated end of the question (`t_uoff`) to the
first agent onset that the reference VAD detects (`t_aon`). A negative value means Moshi
began before the user finished. Most of these came in the pause after "Where is my
order?" ("…I placed it last week."), which is a legitimate full-duplex turn-take that a
cascade would never make. They count in `premature_rate` and pull the mean below the
median. Backchannels would also count as onsets. The session's own `TurnMetrics.voice_to_voice`
starts at the engine's local VAD end of speech and ends at the first forwarded chunk. That
chunk includes the pre-roll and the text token that precedes Moshi's audio (acoustic
delay), so the recording's `residual` is about +200 ms. The recording is the number to
quote.

## Tests

* `tests/test_moshi.py` runs offline against `voice_agent_next.testing.moshi.FakeMoshiServer`.
  The fake speaks the real binary protocol with real Ogg/Opus pages, in the Python and
  Rust handshake flavours. Its output is clocked by input like the real step loop, and
  it yields when talked over, closes at a step limit and handles PersonaPlex prompts. The
  tests drive the engine directly and through an `AgentSession` over `LoopbackTransport`.
* `MOSHI_URL=ws://localhost:8998 pytest -m integration tests/test_moshi.py` runs against
  a real server.
