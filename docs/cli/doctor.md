# `van doctor`

`van doctor` checks whether this machine can run a voice agent and explains what to fix.
The default run only reads information: it does not open the microphone or play sound,
and it makes no network connection except for the local Ollama check that `van presets`
also does. The checks that use the network or your audio devices run only when you pass
their flag.

```bash
van doctor                        # system, audio, hardware, presets, models
van doctor --network              # + DNS/TCP/TLS times of the configured cloud endpoints
van doctor --mic                  # + live microphone level meter (5 s; --duration N)
van doctor --echo                 # + plays a chirp: echo delay and loss, AEC advice
van doctor --latency              # + plays 5 clicks: acoustic loopback latency
van doctor --only audio --json    # one section, machine-readable
```

## Sections

| Section | What it checks | Warnings you may see |
|---|---|---|
| `system` | voice-agent-next and Python versions, OS, ML runtimes (`numpy`, `soxr`, `sounddevice`, `onnxruntime`, `torch`, `mlx`, `aiortc`, `livekit`), accelerators, which API keys are set (names only, never values) | a runtime that is installed but fails to import |
| `audio` | PortAudio version, every host API with its device count, the Linux sound server (PipeWire/PulseAudio socket), default input and output (rate, reported low latency), the rates each default device accepts (8, 16, 22.05, 24, 44.1, 48 kHz) | see below |
| `hardware` | GPUs, CUDA libraries, where `device="auto"` runs local models ([hardware](../hardware.md)) | how to use the GPU when a CPU build is installed |
| `presets` | the readiness check of every [preset](../presets.md) and the one `van run` would pick | no preset ready (`van presets <name>` lists the fixes) |
| `models` | model cache and Hugging Face cache locations, catalog models cached ([models](../models.md)) | partial downloads |
| `network` | opt-in, see [below](#network-network) | unreachable or slow endpoints |
| `mic`, `echo`, `latency` | opt-in, see [below](#interactive-checks) | |

### Audio warnings

| Warning | Why it matters | Fix |
|---|---|---|
| Linux PortAudio without a PulseAudio/PipeWire host API (Debian/Ubuntu ship PortAudio 19.6) | audio goes through ALSA; a raw `hw:` device can be busy | use the ALSA `pipewire`, `pulse` or `default` device ([local audio](../transports/local.md)) |
| raw ALSA `hw:` device as default | exclusive access, fixed rates | the same |
| Windows default devices on MME | 50-100 ms of extra latency | the `Windows WASAPI` variant of the device |
| WDM-KS or ASIO device | exclusive mode: other apps lose the device, and it fails if another app holds it exclusively | the WASAPI variant |
| Bluetooth headset in hands-free mode (HFP/HSP) | 8/16 kHz narrowband audio in both directions hurts recognition and voice quality | a wired/USB headset, or the Bluetooth headset for output only (A2DP) plus another microphone |
| device rate below 16 kHz | speech models expect at least 16 kHz | another device or profile |
| reported latency above 100 ms | every turn waits for it | a lower-latency host API or a wired device |
| macOS without Core Audio, Windows without WASAPI | a broken PortAudio build | reinstall `sounddevice` |

Different default input and output rates are fine (the session resamples) and are only
reported.

## Network (`--network`)

For every cloud provider whose API key is set, `van doctor --network` resolves the host,
opens a TCP connection and completes a TLS handshake, and reports each time. It sends no
request, so it makes no API call and costs nothing. The TCP connect time is about one
network round trip; above 150 ms it is flagged because every streaming turn pays it.

Providers with regional endpoints (AssemblyAI `us`/`eu`, ElevenLabs `us`) are probed in
every region, and the fastest one is suggested as the provider's `region=` option. The
Azure OpenAI resource from `AZURE_OPENAI_ENDPOINT` is probed too. Add other hosts, such
as a self-hosted server, with `--endpoint URL` (repeatable; it implies `--network`).
`--timeout` bounds each probe (5 s by default).

## Interactive checks

These use your devices: the default ones, or `--input-device` / `--output-device` (index
or name, as in [`van devices`](../transports/local.md)). Nothing is recorded to disk.

**`--mic`** records `--duration` seconds (default 5) while a level bar shows the live
level. Speak normally, with pauses. Levels are in dBFS (0 = full scale), measured in 50 ms
blocks:

| Verdict | Rule | Status |
|---|---|---|
| silence | peak ≤ -70 dBFS: muted, wrong device, or the OS blocks microphone access | fail |
| clipping | more than 0.1% of the samples at full scale | warn |
| noisy | noise floor (10th percentile of block levels) above -45 dBFS | warn |
| quiet | loud parts (95th percentile) below -40 dBFS | warn |
| no speech detected | loud parts less than 10 dB above the noise floor | warn |
| DC offset | mean above 2% of full scale | warn |
| good level | none of the above | ok |

**`--echo`** plays a 0.5 s sine sweep (300 Hz to 8 kHz, -6 dBFS) at your normal volume and
records the microphone in the same duplex stream. Cross-correlation finds the round-trip
delay (output latency + air + input latency). The echo return loss (ERL) is the played
level minus the level of the echo at the microphone. From these it recommends
`LocalAudioTransport` echo settings ([audio processing](../audio-processing.md)):

| Measurement | Recommendation |
|---|---|
| no echo above the noise floor | `echo_mode="headphones"` is safe; keep `"auto"` for speakers |
| echo, delay ≤ 500 ms | `echo_mode="aec"` (WebRTC AEC3, extra `aec`); `WebRTCAudioProcessor(delay_ms=<delay>)` gives AEC3 a head start |
| ERL below 6 dB | the same, but lower the speaker volume or move the microphone away first |
| delay above 500 ms (beyond AEC3's range, e.g. Bluetooth) | `echo_mode="half_duplex"` with `half_duplex_tail=<delay + 0.3 s>`, or headphones |

**`--latency`** plays five 40 ms clicks, 0.5 s apart, and reports the mean, minimum and
maximum round trip and the jitter (standard deviation), next to the latency PortAudio
reports for the stream. A loopback cable from output to input measures the audio stack
without the room. Above 250 ms or with more than 10 ms of jitter, the result is a warning.

If the input and output devices do not share a sample rate, the duplex stream is retried at
the input device's rate, then at 48 and 16 kHz.

## Output and exit codes

| Option | Meaning |
|---|---|
| `--json` | print one JSON document instead of tables |
| `--only SECTION` | run only these sections (repeatable); `--network/--mic/--echo/--latency` add theirs |
| `--strict` | exit with 1 on warnings too |

| Exit code | Meaning |
|---|---|
| 0 | no check failed (warnings allowed unless `--strict`) |
| 1 | at least one check failed (with `--strict`: or warned) |
| 2 | invalid options, e.g. an unknown `--only` section |

The JSON document (`"schema": "van-doctor/1"`):

```json
{
  "schema": "van-doctor/1",
  "version": "0.1.0",
  "platform": "linux",
  "exit_code": 0,
  "summary": {"ok": 21, "info": 17, "warn": 1, "fail": 0, "skip": 0},
  "sections": ["system", "audio", "hardware", "presets", "models"],
  "checks": [
    {
      "section": "audio",
      "name": "default input",
      "status": "ok",
      "value": "[4] pipewire (ALSA), 44100 Hz, 9 ms low latency",
      "hint": null,
      "data": {"index": 4, "name": "pipewire", "hostapi": "ALSA",
               "default_samplerate": 44100.0, "low_latency_ms": 8.7,
               "supported_rates": [8000, 16000, 22050, 24000, 44100, 48000]}
    }
  ]
}
```

`status` is one of `ok`, `info`, `warn`, `fail` or `skip`. `hint` says how to fix the
problem. `data` holds the measured values: for example `delay_ms`, `erl_db` and
`recommended` for the echo test, `delays_ms`, `mean_ms` and `jitter_ms` for the latency
probe, and `dns_ms`, `connect_ms` and `tls_ms` for each endpoint.

## From Python

The checks live in `voice_agent_next.doctor`. Each section is a function that returns
`Check` rows, and the analysis functions work on any numpy signal:

```python
from voice_agent_next import doctor

report = doctor.DoctorReport()
report.extend(doctor.audio_checks())
report.extend(doctor.network_checks(doctor.cloud_endpoints()))
print(report.to_dict()["summary"])

stats = doctor.level_stats(samples, 16_000)  # an int16 recording
for status, verdict, fix in doctor.level_verdicts(stats):
    print(status, verdict, fix)
```
