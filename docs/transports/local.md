# Local audio transport (microphone + speakers)

`LocalAudioTransport` connects an agent to this computer's microphone and speakers on
Linux, macOS and Windows. It is built on [sounddevice](https://python-sounddevice.readthedocs.io)
(PortAudio), which is imported only when you create the transport.

```bash
pip install "voice-agent-next[audio]"   # sounddevice + soxr
van devices                             # list devices, host APIs and the defaults
van doctor                              # check PortAudio and the audio setup
van run --engine mock                   # talk to the offline mock engine
```

## Usage

```python
from voice_agent_next import Agent, AgentSession
from voice_agent_next.transports import create_transport
from voice_agent_next.transports.local import LocalAudioTransport

transport = LocalAudioTransport()  # default devices, echo_mode="auto"
transport = LocalAudioTransport(input_device="USB", output_device="USB", echo_mode="headphones")
transport = create_transport({"type": "local", "input_device": 3})  # same options as a dict

await AgentSession("openai/gpt-realtime").run(Agent("You are helpful."), transport)
```

In a config file (`van run -c agent.yaml`):

```yaml
engine: openai/gpt-realtime
transport:
  type: local
  input_device: USB        # index or name, see `van devices`
  output_device: USB
  echo_mode: headphones
```

### Options

| Option | Default | Meaning |
|---|---|---|
| `input_device`, `output_device` | system default | PortAudio index (`3`, `"3"`) or name words (see below) |
| `sample_rate` | device default | Rate of both streams. The session resamples, so you rarely need it. `input_sample_rate`/`output_sample_rate` set one direction |
| `input_channels`, `output_channels` | `1` | Mono by default; PortAudio converts for the device |
| `block_duration` | `None` | Callback period in seconds, e.g. `0.01`. `None` lets the host API choose (`blocksize=0`), the most robust low-latency setting |
| `latency` | `"low"` | PortAudio latency hint: `"low"`, `"high"` or seconds. Try `"high"` if the audio crackles |
| `echo_mode` | `"auto"` | `"auto"`, `"aec"`, `"headphones"`, `"half_duplex"` (see [Echo](#echo)) |
| `echo_canceller` | `None` | An `AudioProcessor` doing echo cancellation |
| `half_duplex_tail` | `0.3` | Seconds the microphone stays muted after the agent stops (half-duplex) |
| `max_buffered` | `1.0` | Seconds of queued agent audio above which `write_audio()` waits |

## Choosing devices

`van devices` prints every device with its index, host API, channel counts, default sample
rate and which devices are the system defaults (`van devices --json` for scripts).

* `None`: the system default input or output device.
* An integer, or a string of digits: that PortAudio index. Indexes can change when devices
  are plugged in or removed; names are more stable.
* A string: space-separated words matched case-insensitively, in order, against
  `"<device name>, <host API>"`, as sounddevice does. `"usb"` matches `USB Audio (hw:1,0)`;
  `"Microphone WASAPI"` matches `Microphone (Realtek(R) Audio), Windows WASAPI`.

When a name matches several devices, an exact name match wins, then the device on the
default device's host API. Otherwise the transport raises `ConfigurationError` listing the
candidates; add a host API word or use the index. `find_audio_device(query, "input")` and
`list_audio_devices()` in `voice_agent_next.transports.local` do the same from Python.

Each stream runs at its device's default sample rate unless you set one. The default rate
always works; other rates depend on the device and host API.

## Echo

Through speakers the microphone hears the agent. Without echo handling, voice activity
detection fires on the agent's own voice and the agent interrupts itself.

| `echo_mode` | What happens | Barge-in |
|---|---|---|
| `"aec"` | Echo cancellation: `echo_canceller`, or the library's default one when installed | yes |
| `"headphones"` | No echo handling. Use it with headphones, or when echo is cancelled outside the app (a speakerphone with built-in echo cancellation, an OS echo-cancelled device) | yes |
| `"half_duplex"` | The microphone is muted (replaced by silence) while the agent is heard and for `half_duplex_tail` seconds after | no |
| `"auto"` (default) | `"aec"` if an echo canceller is available, otherwise `"half_duplex"` with a warning | depends |

How the echo canceller is connected:

* The output callback passes exactly what goes to the speakers, silence included, to
  `echo_canceller.process_render()`. The frame `timestamp` is when it should reach the
  speaker (callback time + output latency).
* Every microphone block goes through `echo_canceller.process_capture()` before it reaches
  the session.
* `process_render()` runs on the output callback thread and `process_capture()` on the input
  callback thread. The transport serializes the calls with a lock, so the processor does
  not need to be thread-safe. Both must be fast: they run on the audio path.
* `reset()` is called on `start()`. The transport calls `close()` only on an echo canceller
  it created itself.
* If the echo canceller raises, the transport logs the error and switches to half-duplex.

Give the echo canceller to the transport, not to `AgentSession(processors=...)`: only the
transport knows when audio is actually played. `transport.input_latency` and
`transport.output_latency` (PortAudio's figures for the open streams) help estimate the
echo delay.

Half-duplex mutes the microphone from the moment agent audio is handed to the device until
`half_duplex_tail` seconds after it has been played. Muted blocks are replaced by silence
of the same length, so the engine's input stream stays continuous. Pausing playback
reopens the microphone once the tail has passed.

## Timing and playback

* **Capture timestamps.** Each microphone frame's `timestamp` is the estimated capture time
  of its first sample: `now() - block duration - input latency`. Latency metrics
  (voice-to-voice) start from it.
* **Playback.** `write_audio()` copies audio into a preallocated ring buffer that the output
  callback drains. The callback never waits for the event loop; when the buffer runs dry
  it plays silence. `write_audio()` waits (back-pressure) while `max_buffered` seconds are
  queued.
* **`clear_audio()`** drops all queued audio immediately (barge-in) and ends a pause.
  Audio already inside the device buffer, at most about the output latency, still plays.
* **`pause_audio()` / `resume_audio()`** stop and restart playback, keeping queued audio.
  The speakers play silence meanwhile.
* **`buffered_duration()`** is the queued audio plus what the device has not played yet
  (output latency). `wait_for_playout()` waits until it reaches zero.
* **Counters.** `input_overflows` and `output_underflows` count blocks PortAudio flagged.
  `dropped_input_frames` counts microphone audio dropped because nobody read
  `audio_input()` (the transport keeps at most 5 s).
* **Errors.** If a device stops (unplugged, for example), `audio_input()` raises
  `TransportError` and the session closes with an error. Hot-plug re-open is not
  supported yet.

## Per-OS notes

### Linux

* **PortAudio.** sounddevice uses the system PortAudio library. Without it, creating the
  transport raises `MissingDependencyError` saying how to install it:
  `sudo apt install libportaudio2` (Debian/Ubuntu), `sudo dnf install portaudio` (Fedora),
  `sudo pacman -S portaudio` (Arch).
* **PipeWire / PulseAudio.** Debian and Ubuntu ship PortAudio 19.6, which only has the
  ALSA and OSS host APIs (JACK too on some builds); the PulseAudio host API is only in
  recent PortAudio development versions. PortAudio therefore reaches PipeWire or PulseAudio through ALSA's
  `default` device and the `pipewire` or `pulse` devices added by `pipewire-alsa` or the
  PulseAudio ALSA plugin. Use one of them (the default device usually is one). Raw
  `hw:X,Y` devices bypass the sound server and may be busy or reject the format.
  `van doctor` detects this setup and says which of these devices exist.
* **Echo cancellation.** PipeWire has a WebRTC-based `echo-cancel` module that creates
  echo-cancelled devices. It needs user session configuration. We have not tested it with
  this transport; if you make those devices the defaults, use `echo_mode="headphones"` so
  echo is not handled twice.

### macOS

* **PortAudio** is bundled in the sounddevice wheels (universal2): nothing else to install.
  The host API is Core Audio.
* **Microphone permission.** macOS asks, on first use, whether the app running Python (your
  terminal or IDE) may use the microphone. If access is denied the microphone delivers
  only silence. Allow it in System Settings › Privacy & Security › Microphone.
* **Bluetooth headsets.** Using the headset's microphone switches it to a
  telephony profile with lower audio quality. For better quality, use the headset for
  output only and the built-in microphone for input (`echo_mode="headphones"` then applies).
* The OS voice-processing echo canceller is not used by PortAudio. Core Audio's
  `change_device_parameters` option is not exposed yet.

### Windows

* **PortAudio** is bundled in the sounddevice wheels, without ASIO unless the
  `SD_ENABLE_ASIO` environment variable is set before sounddevice is imported.
* **Host APIs.** Every device appears once per host API: MME, Windows DirectSound,
  Windows WASAPI and Windows WDM-KS. The system default devices are the MME ones, which
  add latency; `van doctor` points this out. For lower latency pick the WASAPI variant by
  name, e.g. `input_device="Microphone WASAPI"`, `output_device="Speakers WASAPI"`.
  MME may truncate long device names, so match with a short part of the name.
* **WASAPI shared mode** runs at the device's mix-format sample rate, which is the default
  rate the transport uses. Another `sample_rate` on a WASAPI device usually fails to open,
  so leave it unset. Exclusive mode is not exposed yet.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `PortAudio library not found` | Install PortAudio (Linux), or reinstall sounddevice (`pip install --force-reinstall sounddevice`) |
| `no input device matches ...` / `matches several ... devices` | Run `van devices`; use the index or add host API words |
| `cannot open the input device ...` | Leave `sample_rate` unset. On Linux use the `default`/`pipewire` device rather than `hw:` |
| The microphone records only silence | Check OS microphone permission (macOS), the selected device, and its input volume |
| The agent interrupts itself | Use headphones with `echo_mode="headphones"`, `echo_mode="half_duplex"`, or an echo canceller |
| The agent cannot be interrupted | Expected in half-duplex mode (the `"auto"` fallback without echo canceller) |
| Crackling or dropouts (`output_underflows` grows) | `latency="high"` or `block_duration=0.02`; close CPU-heavy programs |

## Testing

The unit tests (`tests/test_transport_local.py`) replace `sounddevice` with a fake module
whose streams invoke the transport's callbacks from their own threads, so they need no
hardware and run in CI on every OS. One smoke test uses real devices: it records 1 s from
the default microphone and plays a short 440 Hz tone. It is marked `audio_device` and
deselected by default:

```bash
uv sync --extra audio
uv run pytest -m audio_device tests/test_transport_local.py
```
