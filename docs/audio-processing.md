# Audio processing: echo cancellation, noise suppression, half-duplex

When an agent talks through loudspeakers, the microphone picks the agent up again. Without
echo control the VAD fires on that echo and the agent interrupts itself. The processors in
`voice_agent_next.audio.aec` fix this. They are `AudioProcessor`s
(`voice_agent_next/audio/processing.py`), so any transport or `AgentSession(processors=...)`
can use them.

| Strategy | Class / mode | Barge-in | Needs |
| --- | --- | --- | --- |
| WebRTC audio processing: AEC3 echo cancellation, noise suppression, high-pass filter, gain control | `WebRTCAudioProcessor`, mode `"aec"` | yes | `pip install 'voice-agent-next[aec]'` (the `livekit` wheel) |
| Mute the mic while the agent is audible | `HalfDuplexGate`, mode `"half_duplex"` | **no** | nothing |
| Headphones: there is no echo path | mode `"headphones"` / `"none"` (no processor) | yes | headphones |
| OS-level echo cancellation (PipeWire `module-echo-cancel`, macOS voice processing, Windows communications mode) | not built in; configure the OS and use `"none"` | yes | OS setup |

```python
from voice_agent_next.audio.aec import create_echo_canceller

# WebRTC APM when the `aec` extra is installed, else None + a warning
echo = create_echo_canceller("auto")
```

`create_echo_canceller(mode)` accepts `"auto"`, `"aec"` (alias `"webrtc"`, raises
`MissingDependencyError` without the extra), `"half_duplex"`, `"headphones"` and `"none"`. Its
keyword arguments configure the processor it creates (see [Options](#options)).

## Wiring: the reference signal

Echo cancellation subtracts what the speakers play from what the microphone hears, so the
processor needs **the audio that is actually played**: its reference, fed through
`process_render(frame)`. Microphone audio goes through `process_capture(frame)`, which returns
the cleaned frame.

* **Feed the reference from exactly one place.** Feeding it from both the transport and the
  session doubles it and breaks the canceller.
* **Prefer the transport's playback callback** (`reference="played"`, the default). The
  callback knows when audio really plays. Feed every block it plays, silence included: a
  continuous reference gave the best results in our tests (about 60-70 dB on every agent
  turn once AEC3 had converged in the first second). When no reference arrives, the
  processor feeds silence itself, so feeding only speech also works.
* **`AgentSession(processors=[...])`** calls `process_render()` when it *queues* audio, up to
  `SessionOptions.output_lookahead` (150 ms) before it plays. Create that processor with
  `reference="queued"`: it then releases the reference in step with the microphone, like a
  playing device. With `"played"` semantics, a reference that runs 150 ms ahead makes AEC3
  flush its buffer and then lose its delay estimate at every pause. In our tests that let
  the echo of the last ~0.5 s of each turn through (down to 2 dB of reduction, against
  55 dB or more with `"queued"`).
* **Report device latencies** when you know them. After opening its streams, a transport
  can call `processor.set_device_latency(input_latency=..., output_latency=...)` (seconds,
  e.g. `sounddevice` `stream.latency`). The APM's stream-delay hint then becomes their sum,
  like LiveKit's console mode. `delay_ms=` overrides it. AEC3 finds the real delay by
  itself (tested up to 400 ms). A good hint only speeds up the first second, and a hint
  much too large slows it down.

`process_render()` only appends the frame to a queue. It is cheap, never waits for audio
processing and is safe to call from a real-time playback callback on another thread. The
APM runs on the capture path: queued reference audio is handed to it right before the
microphone audio that can contain its echo. At most 1 s of reference is kept while the
microphone is not being processed.

## Options

`WebRTCAudioProcessor(...)` and `create_echo_canceller(...)` accept:

| Option | Default | Meaning |
| --- | --- | --- |
| `echo_cancellation` | `True` | AEC3 echo cancellation (needs the reference) |
| `noise_suppression` | `True` | stationary noise suppression (see below before enabling it twice) |
| `high_pass_filter` | `True` | removes DC offset and low-frequency rumble |
| `auto_gain_control` | `True` | normalizes the microphone level |
| `reference` | `"played"` | `"played"`: fed at playback time; `"queued"`: fed ahead of playback (AgentSession) |
| `delay_ms` | `None` | fixed stream-delay hint, 0-500 ms; `None` = reported device latencies or no hint |
| `processing_rate` | `None` | force the APM rate (8000/16000/32000/48000); `None` = the capture rate when possible |
| `tail` (factory, half-duplex) | `0.3` | seconds the gate stays closed after the agent's audio ends |

If every feature is off, `process_capture()` returns its input unchanged.

### Never run noise suppression (or gain control) twice

Stacked noise suppressors distort speech. Transcription and turn detection get worse, and the
VAD sees chopped audio. Turn `noise_suppression` (and `auto_gain_control`) off when something
upstream or downstream already does it, for example:

* browser clients (`getUserMedia` with `noiseSuppression` / `autoGainControl`, the default
  in WebRTC), and phone networks;
* OS voice processing (macOS voice processing I/O, Windows "communications" effects, the
  PipeWire echo-cancel node);
* engines with server-side noise reduction (e.g. OpenAI Realtime
  `input_audio_noise_reduction`) or a dedicated suppressor (RNNoise, DeepFilterNet, Krisp).

## Framing, resampling and latency

The native module takes **exactly 10 ms frames**; any other size aborts the process with a
Rust panic, which Python cannot catch. The processor accepts frames of any size, rate and
channel count and guarantees the 10 ms contract:

* capture audio is re-chunked with `FrameChunker` into 10 ms frames at its own rate. The APM
  accepts any rate that is a multiple of 100 Hz (8-96 kHz verified) and converts internally.
  Other rates (22.05 and 11.025 kHz) are resampled to 48 or 16 kHz and back.
* the returned frame has the **same format and duration** as the input. The added latency is
  the framing remainder: **0** when frames are whole multiples of 10 ms, otherwise just under
  10 ms. It is set once, from the first frame, so no gap is ever inserted mid-stream. Rates
  that need resampling add about 1 ms per resampling stage. `processor.latency` reports the
  framing delay, and it is subtracted from the returned frame's `timestamp`, so latency
  metrics stay exact.
* the reference is converted to mono at the capture processing rate.
* resampling uses the numpy polyphase backend of `Resampler`, which adds 0.5-1 ms of group
  delay. soxr's streaming mode was measured to withhold 20-60 ms of audio. That would blow
  the 10 ms budget, and on the reference path it can make the reference arrive after its echo.
* stereo microphones are processed in stereo.

Cost: about 1% of one CPU core (0.1-0.25 ms per 10-20 ms frame, with all four features on)
at 16, 24, 44.1 and 48 kHz. That is cheap enough to run on the event loop.

## Half-duplex gate

`HalfDuplexGate(tail=0.3, threshold_db=-60.0)` returns silence from `process_capture()` while
the agent is audible and for `tail` seconds afterwards. It needs no dependency and cannot
leak echo, but the user **cannot interrupt** the agent while it speaks.

* Reference frames count as queued back to back. Real-time feeding (a playback callback) and
  bursts ahead of playback (the session's look-ahead) both keep the gate closed until the
  last audible frame has played.
* Frames at or below `threshold_db` (silence, dither) do not close the gate.
* `tail` must cover output latency, input latency and room reverberation. Raise it for
  Bluetooth speakers, whose output latency is often 100-300 ms. `reset()` reopens the gate
  immediately.

## Measured performance

Offline test with synthetic echo (`tests/test_aec.py`, marked `model`). The agent signal is
voiced "words": harmonics of `synth_speech`, gated on and off. AEC3 does not adapt on pure
tones, which it treats as narrow-band noise. The echo is the agent signal delayed, at -6 dB,
with one reflection. Near-end speech is a different voice. Noise suppression and AGC are off,
so only the echo canceller is measured.

| Capture | Echo delay | Echo reduction after convergence | Near-end loss (agent silent) | Double-talk loss |
| --- | --- | --- | --- | --- |
| 16 kHz, 20 ms frames | 40 ms | 42.5 dB | 0.0 dB | 0.7 dB |
| 48 kHz, 7 ms frames | 60 ms | 42.4 dB | 0.3 dB | 1.2 dB |
| 44.1 kHz, 512-sample frames | 120 ms | 31.3 dB | 0.0 dB | 2.8 dB |
| 22.05 kHz (resampled), 331-sample frames | 60 ms | 34.3 dB | 0.3 dB | 1.7 dB |

Without echo cancellation the same pipeline reduces the echo by less than 1 dB. The tests
require at least 15 dB. Session-style feeding with `reference="queued"` (150 ms look-ahead,
two agent turns) stays above 15 dB in every 0.5 s window of the second turn, end included.

Run them with:

```bash
uv sync --extra aec
uv run pytest -q -m model tests/test_aec.py
```

## Platform notes

* The `livekit` wheel ships the native WebRTC library for Linux x86_64/aarch64 (glibc 2.28+),
  macOS x86_64 (10.15+) and arm64 (11+), and Windows x64. There are no wheels for Windows on
  ARM or musl (Alpine). Use `"half_duplex"`, headphones or OS-level echo cancellation there.
* The processor creates the native module lazily, on the first frame. The livekit runtime
  cannot be used across `fork()`, so create and use processors in the same process (e.g.
  inside each worker).
* If the native module fails to load or to process audio, the processor logs the error once
  and passes microphone audio through unprocessed instead of stopping the session.

Alternatives considered in research note 04 §6 but not shipped: `pyaec` (SpeexDSP echo
cancellation, wheels on all three OSes), `aec-audio-processing` (Windows wheel only), and
RNNoise / DeepFilterNet (noise suppression only).
