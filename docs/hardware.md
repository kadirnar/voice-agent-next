# Hardware and backends

Local models (faster-whisper, Kokoro, Silero VAD, Smart Turn) run on CPU everywhere and on
a GPU when one is installed and usable. `voice_agent_next.hardware` decides where each model
runs when you leave `device="auto"` (the default), loads the CUDA libraries the runtimes
need, and explains its choice. `van doctor` prints the same information.

## Quick start

| Machine | Install | What `"auto"` gives |
|---|---|---|
| Linux x86_64 / Windows x64 with an NVIDIA GPU | `pip install 'voice-agent-next[local,cuda]'` | faster-whisper on CUDA (float16); ONNX models on CPU |
| same, with Kokoro on the GPU too | also `pip uninstall -y onnxruntime && pip install 'onnxruntime-gpu[cuda,cudnn]'` | Kokoro on CUDA as well |
| Windows with any DirectX 12 GPU | `pip uninstall -y onnxruntime && pip install onnxruntime-directml` | Kokoro on DirectML |
| Apple silicon | `pip install 'voice-agent-next[local]'` | faster-whisper CPU int8; Kokoro on CoreML |
| anything else | `pip install 'voice-agent-next[local]'` | CPU (int8 for faster-whisper) |

Then check:

```console
$ van doctor
...
│ NVIDIA GPU                       │ NVIDIA GeForce RTX 5070 Ti (16 GB, compute capability 12.0); driver 615.71.09 (CUDA 13.4) │
│ ctranslate2                      │ 4.8.2, 1 CUDA device(s) (bfloat16, float16, float32, int8, ...)                           │
│ CUDA 12 libraries (ctranslate2)  │ cuBLAS 12: pip (nvidia-cublas-cu12)                                                       │
│ faster-whisper device=auto       │ cuda float16 (NVIDIA GeForce RTX 5070 Ti)                                                 │
│ onnxruntime build                │ onnxruntime 1.30.0                                                                        │
│ onnxruntime device=auto (Kokoro) │ cpu (NVIDIA GeForce RTX 5070 Ti found but onnxruntime 1.30.0 has no                       │
│                                  │ CUDAExecutionProvider); to use the GPU: pip uninstall -y onnxruntime && pip install       │
│                                  │ 'onnxruntime-gpu[cuda,cudnn]'                                                             │
```

A yellow row means a faster backend is one install away, and the row gives the command.

## What runs where

| Model | Runtime | `"auto"` | Why |
|---|---|---|---|
| faster-whisper | CTranslate2 | CUDA float16 if usable, else CPU int8 | The final transcription is on the critical path of every turn: ~390 → ~50 ms (see below) |
| Kokoro | ONNX Runtime | CUDA > CoreML > DirectML > CPU, when the installed ONNX Runtime build has them | Unchanged order; CUDA libraries are now loaded first and checked |
| Silero VAD | ONNX Runtime | CPU (`device="cpu"`) | A 32 ms window every 32 ms on a tiny model: a GPU round trip costs more than the inference |
| Smart Turn | ONNX Runtime | CPU (`providers` default) | Runs once per pause, concurrently with the STT flush, so it is off the critical path |
| Chatterbox, Qwen3-TTS | PyTorch | CUDA > MPS > CPU | Autoregressive models of 110M–1.7B parameters: real time needs a GPU ([chatterbox](providers/chatterbox.md), [qwen-tts](providers/qwen-tts.md)) |

Explicit choices are never second-guessed: `device="cpu"` / `device="cuda"` for
faster-whisper and `providers=[...]` for the ONNX models are used as given. With an explicit
`device="cuda"`, a GPU failure is a `ProviderError`, not a fallback.

### Fallbacks

`device="auto"` for faster-whisper:

1. CTranslate2 sees no CUDA device → CPU. With an NVIDIA GPU present the reason is logged
   (`CUDA_VISIBLE_DEVICES` hides it, the driver is too old for CUDA 12...) is shown by
   `van doctor`; a too-old driver is also logged at INFO.
2. CTranslate2 sees the GPU, but cuBLAS 12 does not load → CPU, and one INFO line names the
   fix: `pip install 'voice-agent-next[cuda]'`. The GPU is never tried, so no error is
   raised mid-inference.
3. The libraries load, but loading or the warm-up inference fails on the GPU (no kernels
   for this architecture, out of memory) → a WARNING with the error, and CPU.

For the PyTorch models, `select_torch_backend()` picks CUDA only when torch sees the GPU
*and* was compiled for its architecture: torch < 2.7 has no kernels for Blackwell GPUs
(RTX 50xx, compute capability 12.0) and would fail at the first kernel. Otherwise the model
runs on CPU and the reason and the fix (a CUDA build of torch) are logged and shown by
`van doctor` ("torch device=auto").

For the ONNX models, a CUDA execution provider whose libraries (cuBLAS, cuDNN, cuFFT,
cuRAND, CUDA runtime) do not load is skipped, and when session creation still fails
Kokoro logs a warning and retries on CPU.

## CUDA libraries without `LD_LIBRARY_PATH`

CTranslate2 and ONNX Runtime open the CUDA libraries by file name at run time
(`libcublas.so.12`, `cublas64_12.dll`...) and their wheels do not ship them. NVIDIA
publishes them as pip wheels (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`...), but those
install into `site-packages/nvidia/<component>/lib` (`bin` on Windows), which no loader
searches.

`hardware.load_cuda_libraries()` finds them there and loads them before the runtime needs
them:

* **Linux:** `ctypes.CDLL(path, RTLD_GLOBAL)`, dependencies first (`libcublasLt` before
  `libcublas`). A later `dlopen("libcublas.so.12")` by CTranslate2 resolves to the already
  loaded library by its soname.
* **Windows:** `os.add_dll_directory()` on the wheel's `bin` directory, then the DLLs
  themselves (cuDNN's sub-libraries first).
* Layouts: CUDA 12 wheels (`nvidia/cublas/lib`), CUDA 13 wheels (`nvidia/cu13/lib`, and
  `bin/x86_64` on Windows).
* Libraries not found in `site-packages` are loaded by name from the system library path,
  so a CUDA toolkit on `LD_LIBRARY_PATH` / `PATH` keeps working.

Loading is done once per process, is thread-safe, and never raises: each
`hardware.CudaLibrary` reports `loaded`, `source` (`pip`, `system`, `missing`) and `error`.

The `cuda` extra installs only what CTranslate2 needs (cuBLAS 12, with Blackwell kernels
from 12.8), on Linux x86_64 and Windows x64. It is not part of `local`: it is ~1 GB installed and
useless without an NVIDIA GPU. ONNX Runtime's CUDA build is a different distribution of the
same `onnxruntime` module, so it cannot be an extra: uninstall `onnxruntime` first (two
builds installed side by side overwrite each other; `van doctor` flags it).

## Detection

All detection is cached, import-light (only `ctypes` and the standard library; the ML
runtimes are imported only if installed) and never raises:

| Function | What |
|---|---|
| `detect_nvidia()` | GPUs (name, memory, compute capability), driver version and its CUDA version, through NVML (`libnvidia-ml.so.1` / `nvml.dll`), else `nvidia-smi` |
| `detect_apple_silicon()` | chip name, Rosetta 2 (x86_64 Python on an arm64 Mac: no CoreML/MLX), whether `mlx` is installed |
| `onnxruntime_info()` | version, execution providers, CUDA version of GPU builds, conflicting installed builds |
| `ctranslate2_info()` | version, CUDA device count, CUDA compute types |
| `find_cuda_libraries()` / `load_cuda_libraries()` | see above |
| `torch_info()` | PyTorch version, CUDA version of the build, CUDA devices and their compute capability, the GPU architectures compiled in, MPS |
| `select_ctranslate2_backend()` / `select_onnx_backend()` / `select_torch_backend()` | a `Backend(device, compute_type, providers, reason, fix)` |
| `report()` | the `van doctor` rows |

`clear_cache()` forgets detection results (for tests; loaded libraries stay loaded).

## Measurements

T1 latency, back to back on one machine: AMD Ryzen 5 5600, 32 GB RAM, NVIDIA RTX 5070 Ti
(Blackwell, compute capability 12.0, driver 615.71), Linux 7.2. Silero VAD, Smart Turn v3.2,
faster-whisper `base` (English), Ollama `LiquidAI/lfm2.5-1.2b-instruct` (temperature 0),
Kokoro v1.0 fp16 on CPU. `benchmarks/scenarios/latency-local.yaml` (CPU) and
`latency-local-gpu.yaml` (same stimuli, GPU), 6 turns x 2 sessions, first turn of each
session excluded (10 measured turns):

| faster-whisper | STT final p50 / p90 | end-of-turn delay p50 / p90 | v2v p50 | v2v p90 | TTS first audio p50 |
|---|---:|---:|---:|---:|---:|
| CPU int8 | 391 / 633 ms | 652 / 901 ms | 1,523 ms | 1,750 ms | 614 ms |
| **CUDA float16** | **51 / 68 ms** | **401 / 418 ms** | **969 ms** | 2,975 ms | 439 ms |

* The STT final transcription drops ~8x and the end-of-turn delay reaches its floor: the
  400 ms of silence the VAD waits for. faster-whisper is no longer on the critical path.
* Kokoro still runs on CPU, so TTS first audio depends on the length of the reply's first
  clause, and the LLM does not reply identically across runs even at temperature 0. The GPU
  p90 comes from one prompt ("How long does a refund usually take?") whose GPU-run replies
  opened with a long clause ("Refunds can vary depending on the policy of the company or
  service you ...") that took Kokoro 2.3–3.5 s to render; the CPU run happened to get a
  shorter opening clause. That is a TTS/segmentation effect, not an STT one.
* Blackwell (sm_120) works with CTranslate2 4.8.2 and cuBLAS 12.9 from pip. Loading
  a model on the GPU takes under a second once the driver's kernel cache is warm.

Not measured here: Kokoro on CUDA (needs `onnxruntime-gpu` in place of `onnxruntime`),
CoreML and DirectML. Kokoro's automatic provider order is therefore unchanged.
