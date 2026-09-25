# Energy VAD (`vad: energy`)

A voice activity detector that looks at loudness (RMS energy) only. It has no model and no
dependency beyond numpy, so it is always available, starts instantly and behaves
identically on every platform. It is the VAD of the offline demo, the mock cascades in the
tests and the synthetic benchmarks.

Use it for **clean or synthetic audio**: tests, `van demo`, benchmarks driven by a
simulated caller, file transports. For real microphones and phone calls use
[Silero VAD](silero.md): energy cannot tell speech from other loud sounds (music, a door,
keyboard noise, the agent's own voice leaking back without echo cancellation).

## Usage

```python
from voice_agent_next import AgentSession, create
from voice_agent_next.providers.energy import EnergyVAD

vad = create("vad", "energy")  # default options
vad = EnergyVAD(threshold_db=-35.0, min_silence_duration=0.3)

session = AgentSession(stt=..., llm=..., tts=..., vad="energy")
```

```yaml
# agent.yaml
vad: {provider: energy, threshold_db: -35, min_silence_duration: 0.3}
```

```bash
van run --preset local-cpu --vad energy
van bench latency --stt mock --llm mock --tts mock --vad energy
```

## Options

| Option | Default | Meaning |
|---|---|---|
| `threshold_db` | `-40.0` | Minimum level (dBFS) that counts as speech. |
| `adaptive` | `True` | Track the background noise floor and require speech to rise above it. |
| `margin_db` | `12.0` | With `adaptive`, how far above the noise floor speech must be. |
| `window_ms` | `20.0` | Analysis window. |
| `sample_rate` | `16000` | Analysis rate; input at any rate and channel count is converted by the stream. |
| `options` | `VADOptions()` | Shared thresholds and durations (see [Silero VAD](silero.md#options)). Single fields can be passed as keyword arguments (`min_silence_duration=0.3`). |

## How it works

* Every window's RMS level in dBFS is mapped to a speech probability with a logistic curve
  centred on the threshold (2 dB slope), so the shared `VADOptions` thresholds, minimum
  speech and silence durations and prefix padding work exactly as with Silero.
* With `adaptive`, the noise floor starts at the first window's level (at most
  `threshold_db - margin_db`), falls quickly when the input gets quieter and rises slowly
  when it gets louder. The effective threshold is the higher of `threshold_db` and
  `noise floor + margin_db`, so steady background noise stops counting as speech (the
  floor rises with a time constant of about 10 s at 20 ms windows).
* Each stream (`vad.stream()`) has its own noise floor; `stream.reset()` clears it.

## Performance

About 9 µs per 20 ms window at 16 kHz (≈ 0.05 % of real time) on an AMD Ryzen 5 5600;
`van bench overhead --sections micro` measures it (`energy_vad`) on your machine.

## Limitations

* Loud non-speech is speech to it and quiet speech in a noisy room may be missed; there is
  no spectral or learned model.
* Its onsets and offsets follow the energy envelope, not speech itself, so they differ from
  Silero's on real speech. Compare latency numbers only between runs with the same VAD.
