# Installation

voice-agent-next needs **Python 3.11 or newer** and runs on Linux, macOS and Windows. The
core package is pure Python (numpy, pydantic, httpx, websockets); everything with native
code or a vendor SDK is an **extra** you install only when you use it.

```bash
pip install voice-agent-next              # core: cloud providers over raw WebSocket/HTTP, mock engine, CLI
pip install "voice-agent-next[audio]"     # + microphone and speakers (sounddevice, soxr)
pip install "voice-agent-next[local]"     # + a fully local stack: Silero, Smart Turn, faster-whisper, Kokoro
```

With [uv](https://docs.astral.sh/uv/): `uv add voice-agent-next` in a project, or
`uv tool install "voice-agent-next[audio]"` for the `van` command alone.

## Per operating system

=== "Linux"

    The microphone/speaker transport uses the system PortAudio library:

    ```bash
    sudo apt install libportaudio2        # Debian / Ubuntu
    sudo dnf install portaudio            # Fedora
    sudo pacman -S portaudio              # Arch
    pip install "voice-agent-next[audio]"
    ```

    PortAudio reaches PipeWire or PulseAudio through ALSA's `default`, `pipewire` or
    `pulse` devices; `van doctor` checks this setup (see
    [local audio](../transports/local.md#linux)).

    **NVIDIA GPU** (x86_64): `pip install "voice-agent-next[local,cuda]"` adds the cuBLAS
    12 wheels that faster-whisper needs, loaded from site-packages without
    `LD_LIBRARY_PATH` (see [hardware](../hardware.md)).

=== "macOS"

    PortAudio is bundled in the sounddevice wheels: nothing else to install.

    ```bash
    pip install "voice-agent-next[audio]"
    ```

    The first run asks whether your terminal may use the microphone; if access is denied the
    microphone delivers silence (System Settings › Privacy & Security › Microphone).

    **Apple silicon**: use an arm64 Python. Kokoro runs on CoreML, and the `apple` preset
    uses sherpa-onnx streaming STT and Ollama on Metal. Intel Macs have no PyTorch wheels,
    so the `pocket-tts` extra is skipped there.

=== "Windows"

    PortAudio is bundled in the sounddevice wheels.

    ```powershell
    pip install "voice-agent-next[audio]"
    ```

    Windows lists every device once per host API (MME, DirectSound, WASAPI, WDM-KS). The
    defaults are MME, which adds latency: pick the WASAPI devices by name (`van devices`,
    [local audio](../transports/local.md#windows)).

    **GPUs**: `[local,cuda]` for NVIDIA (faster-whisper on CUDA); any DirectX 12 GPU can run
    Kokoro with `onnxruntime-directml` (see [hardware](../hardware.md)).

## Extras

| Extra | Adds | Used by |
|---|---|---|
| `audio` | sounddevice, soxr | local microphone/speakers transport |
| `resample` | soxr | high-quality streaming resampling (numpy fallback otherwise) |
| `aec` | livekit (WebRTC audio processing) | echo cancellation, noise suppression ([audio processing](../audio-processing.md)) |
| `silero`, `smart-turn`, `onnx` | onnxruntime | Silero VAD, Smart Turn |
| `faster-whisper`, `moonshine`, `sherpa-onnx` | local speech recognition | [faster-whisper](../providers/faster_whisper.md), [Moonshine](../providers/moonshine.md), [sherpa-onnx](../providers/sherpa-onnx.md) |
| `kokoro`, `pocket-tts` | local speech synthesis | [Kokoro](../providers/kokoro.md), [Pocket TTS](../providers/pocket-tts.md) (CPU PyTorch) |
| `openai`, `anthropic`, `google` | vendor SDKs | [OpenAI](../providers/openai.md) (and compatible LLM hosts), [Anthropic](../providers/anthropic.md), [Google](../providers/google.md) |
| `cuda` | cuBLAS 12 wheels | faster-whisper on NVIDIA GPUs |
| `webrtc` | aiortc | [WebRTC transport](../transports/webrtc.md) |
| `otel` | OpenTelemetry API | [tracing](../concepts/observability.md#opentelemetry-tracing) |
| `bench` | jiwer, soundfile | [benchmarks](../benchmarks/methodology.md) (ASR track) |
| `local`, `cloud` | bundles | `local` = audio + silero + smart-turn + faster-whisper + kokoro; `cloud` = openai + anthropic + google |

Deepgram, Cartesia, AssemblyAI, ElevenLabs and the OpenAI Realtime engine talk to their
APIs over raw WebSocket/HTTP and need no extra. The [provider index](../providers/index.md)
lists the extra and environment variables of every provider.

## Check the installation

```bash
van version
van doctor       # Python, audio devices, ML runtimes, GPUs, API keys
van providers    # every provider and what it still needs here
van demo         # offline: a simulated user talks to the mock engine
```

## From source

```bash
git clone https://github.com/kadirnar/voice-agent-next && cd voice-agent-next
uv sync                      # dev tools; add --extra <name> for provider dependencies
uv run pytest -q             # fast unit tests: no network, no model downloads
```

Next: [your first agent](first-agent.md).
