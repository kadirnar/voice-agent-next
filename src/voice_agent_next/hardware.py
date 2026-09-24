"""Hardware detection and backend selection for local models.

Local providers call this module to decide where a model runs when ``device="auto"``,
and ``van doctor`` uses it to explain those decisions (see ``docs/hardware.md``):

* **Detection** is cheap, cached and never raises: NVIDIA GPUs and driver through NVML
  (``ctypes``, with ``nvidia-smi`` as a fallback), Apple silicon, the execution providers
  of the installed ONNX Runtime build, CTranslate2's CUDA devices.
* **CUDA libraries**: CTranslate2 and ONNX Runtime open cuBLAS, cuDNN... by file name at
  run time. :func:`load_cuda_libraries` finds them in NVIDIA's pip wheels
  (``site-packages/nvidia/*/lib``; ``bin`` on Windows) and loads them first, so that
  ``pip install 'voice-agent-next[cuda]'`` is all it takes: no ``LD_LIBRARY_PATH`` or
  ``PATH`` changes.
* **Selection**: :func:`select_ctranslate2_backend` and :func:`select_onnx_backend` return
  a :class:`Backend`: the device, why it was chosen and, when a faster one is an install
  away, the command that enables it.
"""

from __future__ import annotations

import ctypes
import functools
import importlib
import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
import threading
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .errors import ConfigurationError
from .utils.deps import is_installed
from .utils.log import logger

__all__ = [
    "CPU_PROVIDER",
    "CTRANSLATE2_CUDA_LIBRARIES",
    "CUDA_EXTRA_HINT",
    "EXECUTION_PROVIDERS",
    "GPU",
    "ONNXRUNTIME_CUDA_LIBRARIES",
    "ONNXRUNTIME_GPU_HINT",
    "ONNX_DEVICES",
    "TORCH_CUDA_HINT",
    "TORCH_DEVICES",
    "AppleSilicon",
    "Backend",
    "CTranslate2Info",
    "CudaLibrary",
    "NvidiaInfo",
    "OnnxRuntimeInfo",
    "TorchInfo",
    "clear_cache",
    "ctranslate2_compute_type",
    "ctranslate2_info",
    "detect_apple_silicon",
    "detect_nvidia",
    "find_cuda_libraries",
    "load_cuda_libraries",
    "onnxruntime_info",
    "parse_nvidia_smi",
    "report",
    "select_ctranslate2_backend",
    "select_onnx_backend",
    "select_torch_backend",
    "session_providers",
    "torch_info",
]

CUDA_EXTRA_HINT = "pip install 'voice-agent-next[cuda]'"
"""Installs the CUDA libraries CTranslate2 (faster-whisper) needs."""
ONNXRUNTIME_GPU_HINT = "pip uninstall -y onnxruntime && pip install 'onnxruntime-gpu[cuda,cudnn]'"
"""Swaps the CPU build of ONNX Runtime for the CUDA one (the two cannot be installed together)."""
TORCH_CUDA_HINT = (
    "pip install --force-reinstall torch torchaudio "
    "--index-url https://download.pytorch.org/whl/cu130  (NVIDIA driver >= 580; cu128 for older)"
)
"""Replaces a CPU (or too old) PyTorch build with a CUDA 13 one (Blackwell GPUs need >= 2.7)."""

CPU_PROVIDER = "CPUExecutionProvider"
EXECUTION_PROVIDERS: dict[str, str] = {
    "cuda": "CUDAExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
    "directml": "DmlExecutionProvider",
    "cpu": CPU_PROVIDER,
}
"""ONNX Runtime execution provider of each device name."""
ONNX_DEVICES = ("auto", *EXECUTION_PROVIDERS)
"""Valid ``device`` values of ONNX Runtime based providers."""

CTRANSLATE2_CUDA_LIBRARIES = ("cublas",)
"""What CTranslate2 >= 4.6 opens on CUDA (cuDNN became optional in 4.6.3)."""
ONNXRUNTIME_CUDA_LIBRARIES = ("cublas", "cudart", "curand", "cufft", "cudnn")
"""What ONNX Runtime's CUDA execution provider opens."""

_ONNXRUNTIME_DISTRIBUTIONS = (
    "onnxruntime",
    "onnxruntime-gpu",
    "onnxruntime-directml",
    "onnxruntime-openvino",
    "onnxruntime-qnn",
    "onnxruntime-rocm",
    "onnxruntime-migraphx",
)
"""Distributions installing the ``onnxruntime`` module (at most one works at a time)."""

_INSTALL_HINTS = {
    "coreml": "the standard onnxruntime wheel for macOS includes it",
    "directml": "pip uninstall -y onnxruntime && pip install onnxruntime-directml (Windows)",
    "cuda": ONNXRUNTIME_GPU_HINT,
}


# ------------------------------------------------------------------------------ GPUs
@dataclass(frozen=True, slots=True)
class GPU:
    """An NVIDIA GPU seen by the driver."""

    index: int
    name: str
    memory_mib: int | None = None
    compute_capability: tuple[int, int] | None = None

    def __str__(self) -> str:
        details = []
        if self.memory_mib:
            details.append(f"{self.memory_mib / 1024:.0f} GB")
        if self.compute_capability:
            details.append("compute capability {}.{}".format(*self.compute_capability))
        return f"{self.name} ({', '.join(details)})" if details else self.name


@dataclass(frozen=True, slots=True)
class NvidiaInfo:
    """The NVIDIA driver and its GPUs (empty without an NVIDIA driver)."""

    gpus: tuple[GPU, ...] = ()
    driver_version: str | None = None
    cuda_version: tuple[int, int] | None = None
    """Newest CUDA version the driver supports."""
    source: str | None = None
    """``"nvml"`` or ``"nvidia-smi"``."""

    def gpu_name(self, index: int = 0) -> str:
        """Name of GPU ``index`` for messages (``"a CUDA GPU"`` when unknown)."""
        for gpu in self.gpus:
            if gpu.index == index:
                return gpu.name
        return "a CUDA GPU"


class _NvmlMemory(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


def _nvml_candidates() -> list[str]:
    if sys.platform == "win32":
        system32 = Path(os.environ.get("SYSTEMROOT", r"C:\Windows")) / "System32"
        nvsmi = Path(os.environ.get("PROGRAMFILES", r"C:\Program Files")) / "NVIDIA Corporation"
        return [str(system32 / "nvml.dll"), str(nvsmi / "NVSMI" / "nvml.dll")]
    if sys.platform == "darwin":
        return []
    return ["libnvidia-ml.so.1", "libnvidia-ml.so"]


def _query_nvml() -> NvidiaInfo | None:
    """GPUs and driver from NVML (``None`` when NVML is unavailable)."""
    lib: Any = None
    for candidate in _nvml_candidates():
        try:
            lib = ctypes.CDLL(candidate)
            break
        except OSError:
            continue
    if lib is None:
        return None
    try:
        if lib.nvmlInit_v2() != 0:
            return None
    except AttributeError:
        return None
    try:
        text = ctypes.create_string_buffer(96)
        driver = (
            text.value.decode(errors="replace")
            if lib.nvmlSystemGetDriverVersion(text, len(text)) == 0
            else None
        )
        version = ctypes.c_int()
        get_cuda = getattr(lib, "nvmlSystemGetCudaDriverVersion_v2", None) or getattr(
            lib, "nvmlSystemGetCudaDriverVersion", None
        )
        cuda = None
        if get_cuda is not None and get_cuda(ctypes.byref(version)) == 0 and version.value > 0:
            cuda = (version.value // 1000, version.value % 1000 // 10)
        count = ctypes.c_uint()
        if lib.nvmlDeviceGetCount_v2(ctypes.byref(count)) != 0:
            count.value = 0
        gpus = []
        for index in range(count.value):
            handle = ctypes.c_void_p()
            if lib.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(handle)) != 0:
                continue
            name = ctypes.create_string_buffer(96)
            if lib.nvmlDeviceGetName(handle, name, len(name)) != 0:
                continue
            memory = _NvmlMemory()
            memory_mib = (
                int(memory.total // 2**20)
                if lib.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory)) == 0
                else None
            )
            major, minor = ctypes.c_int(), ctypes.c_int()
            capability = (
                (major.value, minor.value)
                if lib.nvmlDeviceGetCudaComputeCapability(
                    handle, ctypes.byref(major), ctypes.byref(minor)
                )
                == 0
                else None
            )
            gpus.append(GPU(index, name.value.decode(errors="replace"), memory_mib, capability))
        return NvidiaInfo(tuple(gpus), driver, cuda, "nvml")
    except (AttributeError, OSError, ValueError):
        return None
    finally:
        try:
            lib.nvmlShutdown()
        except (AttributeError, OSError):
            pass


def parse_nvidia_smi(output: str) -> tuple[GPU, ...]:
    """GPUs from ``nvidia-smi --query-gpu=index,name,memory.total,compute_cap
    --format=csv,noheader,nounits`` output."""
    gpus = []
    for line in output.splitlines():
        index, sep, rest = line.partition(",")
        fields = [f.strip() for f in rest.rsplit(",", 2)] if sep else []
        if len(fields) != 3 or not index.strip().isdigit():
            continue
        name, memory, capability = fields
        major, _, minor = capability.partition(".")
        gpus.append(
            GPU(
                int(index),
                name,
                int(memory) if memory.isdigit() else None,
                (int(major), int(minor)) if major.isdigit() and minor.isdigit() else None,
            )
        )
    return tuple(gpus)


def _query_nvidia_smi() -> NvidiaInfo | None:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    query = "--query-gpu=index,name,memory.total,compute_cap"
    try:
        gpus = subprocess.run(
            [exe, query, "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout  # fmt: skip
        driver = subprocess.run(
            [exe, "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        return None
    lines = driver.split()
    return NvidiaInfo(parse_nvidia_smi(gpus), lines[0] if lines else None, None, "nvidia-smi")


@functools.cache
def detect_nvidia() -> NvidiaInfo:
    """NVIDIA driver and GPUs, through NVML or else ``nvidia-smi`` (cached)."""
    info = _query_nvml() or _query_nvidia_smi() or NvidiaInfo()
    logger.debug("hardware: NVIDIA GPUs %s (driver %s)", info.gpus, info.driver_version)
    return info


@dataclass(frozen=True, slots=True)
class AppleSilicon:
    """An Apple silicon Mac (CoreML and MLX capable)."""

    chip: str | None
    rosetta: bool = False
    """Python is an x86_64 build running under Rosetta 2: CoreML/MLX need an arm64 Python."""
    mlx: bool = False
    """The ``mlx`` package is installed."""

    def __str__(self) -> str:
        text = self.chip or "Apple silicon"
        return f"{text} (x86_64 Python under Rosetta 2)" if self.rosetta else text


def _sysctl(name: str) -> str | None:
    try:
        out = subprocess.run(
            ["sysctl", "-n", name], capture_output=True, text=True, timeout=5, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


@functools.cache
def detect_apple_silicon() -> AppleSilicon | None:
    """The Apple silicon chip, or ``None`` on other machines (cached)."""
    if sys.platform != "darwin":
        return None
    machine = platform.machine()
    rosetta = machine == "x86_64" and _sysctl("sysctl.proc_translated") == "1"
    if machine != "arm64" and not rosetta:
        return None  # an Intel Mac
    return AppleSilicon(_sysctl("machdep.cpu.brand_string"), rosetta, is_installed("mlx"))


# ------------------------------------------------------------------ ML runtimes
@dataclass(frozen=True, slots=True)
class OnnxRuntimeInfo:
    """The installed ONNX Runtime build."""

    version: str
    providers: tuple[str, ...]
    """Execution providers compiled into this build."""
    distributions: tuple[str, ...] = ()
    """Installed distributions of the ``onnxruntime`` module; more than one is broken."""
    cuda_version: str | None = None
    """CUDA version of a GPU build (``"13.0"``)."""

    @property
    def cuda_major(self) -> int | None:
        major = (self.cuda_version or "").split(".")[0]
        return int(major) if major.isdigit() else None

    @property
    def build(self) -> str:
        """``"onnxruntime-gpu 1.30.0 (CUDA 13.0)"``."""
        name = "/".join(self.distributions) or "onnxruntime"
        cuda = f" (CUDA {self.cuda_version})" if self.cuda_version else ""
        return f"{name} {self.version}{cuda}"


def _installed_distributions(names: Sequence[str]) -> tuple[str, ...]:
    found = []
    for name in names:
        try:
            importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        found.append(name)
    return tuple(found)


def onnxruntime_info(ort: Any = None) -> OnnxRuntimeInfo | None:
    """The ONNX Runtime build (``None`` when it is not installed or cannot be imported)."""
    if ort is None:
        if not is_installed("onnxruntime"):
            return None
        try:
            ort = importlib.import_module("onnxruntime")
        except Exception as exc:  # broken native libraries
            logger.debug("hardware: onnxruntime does not import: %s", exc)
            return None
    try:
        providers = tuple(ort.get_available_providers())
    except Exception:  # fakes and very old builds
        providers = (CPU_PROVIDER,)
    cuda = getattr(ort, "cuda_version", None)
    if not isinstance(cuda, str):
        info = sys.modules.get(f"{ort.__name__}.capi.build_and_package_info")
        cuda = getattr(info, "cuda_version", None)
    return OnnxRuntimeInfo(
        version=str(getattr(ort, "__version__", "unknown")),
        providers=providers,
        distributions=_installed_distributions(_ONNXRUNTIME_DISTRIBUTIONS),
        cuda_version=cuda if isinstance(cuda, str) and cuda else None,
    )


@dataclass(frozen=True, slots=True)
class CTranslate2Info:
    """The installed CTranslate2 (the faster-whisper runtime)."""

    version: str
    cuda_devices: int
    cuda_compute_types: tuple[str, ...] = ()

    @property
    def cuda_major(self) -> int | None:
        """CUDA major version of the PyPI wheels of this release (``None``: unknown)."""
        major = self.version.split(".")[0]
        return 12 if major == "4" else None


def _cuda_device_count(ct2: Any) -> int:
    try:
        return int(ct2.get_cuda_device_count())
    except Exception:  # CPU-only builds (macOS, Linux aarch64), broken drivers
        return 0


def ctranslate2_info(ct2: Any = None) -> CTranslate2Info | None:
    """CTranslate2's view of the machine (``None`` when it is not installed)."""
    if ct2 is None:
        if not is_installed("ctranslate2"):
            return None
        try:
            ct2 = importlib.import_module("ctranslate2")
        except Exception as exc:
            logger.debug("hardware: ctranslate2 does not import: %s", exc)
            return None
    devices = _cuda_device_count(ct2)
    types: tuple[str, ...] = ()
    if devices:
        try:
            types = tuple(sorted(ct2.get_supported_compute_types("cuda")))
        except Exception:
            types = ()
    return CTranslate2Info(str(getattr(ct2, "__version__", "unknown")), devices, types)


_COMPUTE_TYPES = {"cuda": ("float16", "int8", "float32"), "cpu": ("int8", "float32")}


def ctranslate2_compute_type(ct2: Any, device: str, requested: str = "auto") -> str:
    """``requested``, or for ``"auto"`` the fastest type CTranslate2 supports on ``device``.

    float16 on CUDA (int8 on GPUs without fast float16), int8 on CPU; ``"default"`` when
    the supported types cannot be queried.
    """
    if requested != "auto":
        return requested
    try:
        supported = set(ct2.get_supported_compute_types(device))
    except Exception:
        supported = set()
    return next((ct for ct in _COMPUTE_TYPES.get(device, ()) if ct in supported), "default")


@dataclass(frozen=True, slots=True)
class TorchInfo:
    """The installed PyTorch build and the devices it can use."""

    version: str
    cuda_version: str | None = None
    """CUDA version of a CUDA build (``"13.0"``); ``None`` for CPU builds."""
    cuda_devices: int = 0
    arch_list: tuple[str, ...] = ()
    """GPU architectures compiled into the build (``"sm_120"``, ``"compute_90"``...)."""
    capabilities: tuple[tuple[int, int], ...] = ()
    """Compute capability of each CUDA device."""
    mps: bool = False
    """Apple Metal (MPS) is usable."""

    def supports(self, capability: tuple[int, int]) -> bool:
        """Whether this build has kernels for a GPU of this compute capability.

        CUDA binaries run on GPUs of the same major version and a newer or equal minor
        version; PTX (``compute_XY``) is compiled at load time for any newer GPU. An empty
        arch list (unknown) counts as supported.
        """
        if not self.arch_list:
            return True
        major, minor = capability
        for arch in self.arch_list:
            kind, _, digits = arch.partition("_")
            if not digits.isdigit() or len(digits) < 2:
                continue
            a_major, a_minor = int(digits[:-1]), int(digits[-1])
            if kind == "sm" and a_major == major and a_minor <= minor:
                return True
            if kind == "compute" and (a_major, a_minor) <= capability:
                return True
        return False


def torch_info(torch: Any = None) -> TorchInfo | None:
    """PyTorch's view of the machine (``None`` when it is not installed or broken)."""
    if torch is None:
        if not is_installed("torch"):
            return None
        try:
            torch = importlib.import_module("torch")
        except Exception as exc:
            logger.debug("hardware: torch does not import: %s", exc)
            return None
    cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    devices = 0
    arch_list: tuple[str, ...] = ()
    capabilities: tuple[tuple[int, int], ...] = ()
    try:
        if torch.cuda.is_available():
            devices = int(torch.cuda.device_count())
            arch_list = tuple(torch.cuda.get_arch_list())
            capabilities = tuple(
                (int(cap[0]), int(cap[1]))
                for cap in (torch.cuda.get_device_capability(i) for i in range(devices))
            )
    except Exception as exc:  # broken drivers
        logger.debug("hardware: torch.cuda failed: %s", exc)
        devices = 0
    try:
        mps = bool(torch.backends.mps.is_available())
    except Exception:
        mps = False
    return TorchInfo(
        version=str(getattr(torch, "__version__", "unknown")),
        cuda_version=cuda_version if isinstance(cuda_version, str) and cuda_version else None,
        cuda_devices=devices,
        arch_list=arch_list,
        capabilities=capabilities,
        mps=mps,
    )


# --------------------------------------------------------------- CUDA libraries
@dataclass(frozen=True, slots=True)
class CudaLibrary:
    """A CUDA library that a backend opens by file name at run time."""

    component: str
    """``"cublas"``, ``"cudart"``, ``"cufft"``, ``"curand"`` or ``"cudnn"``."""
    name: str
    """Display name, e.g. ``"cuBLAS 12"``."""
    files: tuple[str, ...]
    """File names on this OS, in load order (``libcublasLt.so.12``, ``libcublas.so.12``)."""
    package: str
    """The pip package that ships it, e.g. ``nvidia-cublas-cu12``."""
    path: Path | None = None
    """Directory of the pip wheel it is found (or loaded) in."""
    loaded: bool = False
    """Loaded into this process, from :attr:`path` or from the system library path."""
    error: str | None = None
    """Why it could not be loaded."""

    @property
    def found(self) -> bool:
        return self.loaded or self.path is not None

    @property
    def source(self) -> str:
        """``"pip"``, ``"system"`` or ``"missing"``."""
        return "pip" if self.path is not None else "system" if self.loaded else "missing"

    def __str__(self) -> str:
        where = {"pip": f"pip ({self.package})", "system": "system library path"}
        return f"{self.name}: {where.get(self.source, 'missing')}"


_LIBRARY_SPECS: dict[str, tuple[str, tuple[str, ...], tuple[str, ...], str]] = {
    # component: (display name, Linux files, Windows files, pip package), where {v} is the
    # CUDA major version, {fft} the cuFFT one and {cu} the "-cuNN" suffix that CUDA 12
    # package names have (CUDA 13 wheels dropped it, except cuDNN's).
    "cublas": (
        "cuBLAS {v}",
        ("libcublasLt.so.{v}", "libcublas.so.{v}"),
        ("cublasLt64_{v}.dll", "cublas64_{v}.dll"),
        "nvidia-cublas{cu}",
    ),
    "cudart": (
        "CUDA runtime {v}",
        ("libcudart.so.{v}",),
        ("cudart64_{v}.dll",),
        "nvidia-cuda-runtime{cu}",
    ),
    "cufft": ("cuFFT {fft}", ("libcufft.so.{fft}",), ("cufft64_{fft}.dll",), "nvidia-cufft{cu}"),
    "curand": ("cuRAND 10", ("libcurand.so.10",), ("curand64_10.dll",), "nvidia-curand{cu}"),
    "cudnn": ("cuDNN 9", ("libcudnn.so.9",), ("cudnn64_9.dll",), "nvidia-cudnn-cu{v}"),
}


def _library(component: str, cuda_major: int, windows: bool) -> CudaLibrary:
    try:
        name, linux, win, package = _LIBRARY_SPECS[component]
    except KeyError:
        raise ValueError(
            f"unknown CUDA library {component!r}; known: {', '.join(_LIBRARY_SPECS)}"
        ) from None
    fields = {"v": cuda_major, "fft": cuda_major - 1, "cu": f"-cu{cuda_major}" * (cuda_major < 13)}
    return CudaLibrary(
        component,
        name.format(**fields),
        tuple(f.format(**fields) for f in (win if windows else linux)),
        package.format(**fields),
    )


def _wheel_dirs(search_path: Sequence[str], windows: bool) -> list[Path]:
    """Library directories of NVIDIA's pip wheels (``<entry>/nvidia/<package>/lib``).

    CUDA 12 wheels install one directory per component (``nvidia/cublas/lib``), CUDA 13
    wheels share one (``nvidia/cu13/lib``); on Windows the DLLs are in ``bin`` (CUDA 13:
    ``bin/x86_64`` or ``bin/arm64``, the wheel's architecture).
    """
    bins = ("bin", os.path.join("bin", "x86_64"), os.path.join("bin", "arm64"))
    subdirs = bins if windows else ("lib",)
    dirs: list[Path] = []
    seen: set[str] = set()
    for entry in search_path:
        root = Path(entry or os.curdir) / "nvidia"
        try:
            packages = sorted(p for p in root.iterdir() if p.is_dir())
        except OSError:
            continue
        for package in packages:
            for sub in subdirs:
                directory = package / sub
                key = os.path.normcase(os.path.abspath(directory))
                if key not in seen and directory.is_dir():
                    seen.add(key)
                    dirs.append(directory)
    return dirs


def find_cuda_libraries(
    components: Sequence[str],
    cuda_major: int,
    *,
    search_path: Sequence[str] | None = None,
    windows: bool | None = None,
) -> tuple[CudaLibrary, ...]:
    """Locate CUDA libraries in NVIDIA's pip wheels, without loading anything.

    Args:
        components: e.g. :data:`CTRANSLATE2_CUDA_LIBRARIES` (``("cublas",)``).
        cuda_major: CUDA major version the backend was built for (12, 13...).
        search_path: directories holding an ``nvidia`` package (default: ``sys.path``).
        windows: file-name convention (default: this OS).

    Returns:
        One :class:`CudaLibrary` per component; :attr:`CudaLibrary.path` is ``None`` for
        libraries that are not installed with pip.
    """
    windows = sys.platform == "win32" if windows is None else windows
    dirs = _wheel_dirs(sys.path if search_path is None else search_path, windows)
    found = []
    for component in components:
        library = _library(component, cuda_major, windows)
        path = next((d for d in dirs if all((d / f).is_file() for f in library.files)), None)
        found.append(replace(library, path=path))
    return tuple(found)


_load_lock = threading.Lock()
_loaded: dict[tuple[str, int], CudaLibrary] = {}
_handles: list[Any] = []
"""Loaded libraries and DLL-directory cookies: kept alive for the life of the process."""
_dll_dirs: set[str] = set()


def _open_file(path: Path) -> Any:
    """Load a library by path; later loads by file name (dlopen/LoadLibrary) resolve to it."""
    if sys.platform == "win32":
        directory = str(path.parent)
        if directory not in _dll_dirs:  # searched for the dependencies of later loads too
            _dll_dirs.add(directory)
            _handles.append(os.add_dll_directory(directory))
        return ctypes.WinDLL(str(path))
    # Global: symbols resolve for libraries loaded later. CUDA libraries version their
    # symbols (cublasCreate_v2@@libcublas.so.12), so CUDA 12 and 13 can coexist.
    return ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)


def _open_name(name: str) -> Any:
    """Load a library by file name through the platform's default search, as backends do."""
    if sys.platform == "win32":
        return ctypes.WinDLL(name, winmode=0)  # LoadLibrary's standard search, incl. PATH
    return ctypes.CDLL(name, mode=ctypes.RTLD_GLOBAL)


def _companions(library: CudaLibrary) -> list[str]:
    """Extra files to load with ``library``: cuDNN's sub-libraries on Windows.

    The cuDNN 9 shim opens them by name; on Linux it finds them next to itself.
    """
    if (
        library.component != "cudnn"
        or library.path is None
        or not library.files[0].endswith(".dll")
    ):
        return []
    return sorted(p.name for p in library.path.glob("cudnn_*64_9.dll"))


def _load(library: CudaLibrary) -> CudaLibrary:
    errors = []
    if library.path is not None:
        try:
            for name in [*_companions(library), *library.files]:
                _handles.append(_open_file(library.path / name))
            logger.debug("hardware: loaded %s from %s", library.name, library.path)
            return replace(library, loaded=True)
        except OSError as exc:
            errors.append(f"{library.path}: {exc}")
    try:
        for name in library.files:
            _handles.append(_open_name(name))
    except OSError as exc:
        errors.append(str(exc))
        return replace(library, path=None, loaded=False, error="; ".join(errors))
    logger.debug("hardware: loaded %s from the system library path", library.name)
    return replace(library, path=None, loaded=True)


def load_cuda_libraries(components: Sequence[str], cuda_major: int) -> tuple[CudaLibrary, ...]:
    """Load CUDA libraries so that backends opening them by file name find them.

    Each library comes from NVIDIA's pip wheels when installed (``nvidia-cublas-cu12``...,
    see :func:`find_cuda_libraries`), else from the system library path. Loaded libraries
    stay loaded and results are cached per library, so repeated calls are free. Never
    raises: check :attr:`CudaLibrary.loaded`.
    """
    results = []
    with _load_lock:
        for component in components:
            key = (component, cuda_major)
            if key not in _loaded:
                (library,) = find_cuda_libraries((component,), cuda_major)
                _loaded[key] = _load(library)
            results.append(_loaded[key])
    return tuple(results)


def _missing(libraries: Sequence[CudaLibrary]) -> str:
    """``"libcublas.so.12"`` / ``"libcudnn.so.9 and libcufft.so.11"``."""
    names = [lib.files[-1] for lib in libraries if not lib.loaded]
    return " and ".join([", ".join(names[:-1]), names[-1]] if len(names) > 1 else names)


def clear_cache() -> None:
    """Forget cached detection results (loaded libraries stay loaded)."""
    detect_nvidia.cache_clear()
    detect_apple_silicon.cache_clear()
    with _load_lock:
        _loaded.clear()


# ------------------------------------------------------------------ selection
@dataclass(frozen=True, slots=True)
class Backend:
    """Where a local model runs, and why."""

    device: str
    """``"cpu"``, ``"cuda"``, ``"coreml"`` or ``"directml"``."""
    compute_type: str | None = None
    """CTranslate2 compute type (``"float16"``, ``"int8"``...) for CTranslate2 models."""
    providers: tuple[str, ...] = ()
    """ONNX Runtime execution providers, in priority order, for ONNX Runtime models."""
    reason: str = ""
    """Why (one line, for logs and ``van doctor``)."""
    fix: str | None = None
    """Command that enables a faster backend, when one is an install away."""

    def __str__(self) -> str:
        device = f"{self.device} {self.compute_type}" if self.compute_type else self.device
        return f"{device} ({self.reason})" if self.reason else device


def select_ctranslate2_backend(
    device: str = "auto",
    compute_type: str = "auto",
    *,
    ct2: Any = None,
    device_index: int = 0,
) -> Backend:
    """Where CTranslate2 (faster-whisper) runs.

    ``"auto"`` picks CUDA when CTranslate2 sees a GPU *and* the CUDA libraries it opens at
    run time are loadable (they are loaded from pip wheels first, see
    :func:`load_cuda_libraries`), CPU otherwise. ``"cuda"`` and ``"cpu"`` are kept as
    requested (the libraries are still loaded for ``"cuda"``). The compute type follows
    :func:`ctranslate2_compute_type`. Whether CUDA really works is only known once a model
    has run on it: callers should fall back to CPU when that fails.

    Args:
        ct2: the ``ctranslate2`` module (imported when omitted).
    """
    if ct2 is None:
        ct2 = importlib.import_module("ctranslate2")
    info = ctranslate2_info(ct2)
    assert info is not None
    if device == "cpu":
        return Backend(
            "cpu", ctranslate2_compute_type(ct2, "cpu", compute_type), reason="requested"
        )
    libraries: tuple[CudaLibrary, ...] = ()
    if info.cuda_major is not None and (info.cuda_devices or device == "cuda"):
        libraries = load_cuda_libraries(CTRANSLATE2_CUDA_LIBRARIES, info.cuda_major)
    if device != "auto":
        return Backend(
            device, ctranslate2_compute_type(ct2, device, compute_type), reason="requested"
        )

    def cpu(reason: str, fix: str | None = None) -> Backend:
        return Backend("cpu", ctranslate2_compute_type(ct2, "cpu", compute_type), (), reason, fix)

    nvidia = detect_nvidia()
    if not info.cuda_devices:
        if not nvidia.gpus:
            return cpu("no CUDA GPU")
        if os.environ.get("CUDA_VISIBLE_DEVICES", "unset").strip() in ("", "-1"):
            return cpu(f"{nvidia.gpu_name()} hidden by CUDA_VISIBLE_DEVICES")
        needed = info.cuda_major or 12
        if nvidia.cuda_version is not None and nvidia.cuda_version < (needed, 0):
            return cpu(
                f"{nvidia.gpu_name()} found, but NVIDIA driver {nvidia.driver_version} only "
                f"supports CUDA {nvidia.cuda_version[0]}.{nvidia.cuda_version[1]} and "
                f"CTranslate2 {info.version} needs CUDA {needed}",
                "update the NVIDIA driver",
            )
        return cpu(
            f"{nvidia.gpu_name()} found, but CTranslate2 {info.version} has no CUDA support here"
        )
    gpu = nvidia.gpu_name(device_index)
    if any(not lib.loaded for lib in libraries):
        return cpu(f"{gpu} found but {_missing(libraries)} is missing", CUDA_EXTRA_HINT)
    return Backend("cuda", ctranslate2_compute_type(ct2, "cuda", compute_type), reason=gpu)


def select_onnx_backend(
    device: str = "auto",
    *,
    accelerators: Sequence[str] = ("cuda", "directml"),
    cpu_reason: str | None = None,
    ort: Any = None,
) -> Backend:
    """ONNX Runtime execution providers for a model.

    ``"auto"`` walks ``accelerators`` in order and picks the first one whose execution
    provider is compiled into the installed ONNX Runtime and, for CUDA, whose libraries are
    loadable (loaded from pip wheels first); CPU otherwise. The CPU provider always comes
    last, for operators the accelerator does not support. ONNX Runtime silently runs a
    session on CPU when an accelerator fails to initialize: compare the result with
    :func:`session_providers`.

    Args:
        device: ``"auto"``, ``"cpu"``, ``"cuda"``, ``"coreml"`` or ``"directml"``.
        accelerators: devices that make this model faster, best first (``"auto"`` only).
            Leave CoreML out unless measured: models with dynamic shapes are split into
            many CoreML/CPU partitions and get slower.
        cpu_reason: why the model stays on CPU when ``accelerators`` is empty.
        ort: the ``onnxruntime`` module (imported when omitted).

    Raises:
        ConfigurationError: an explicitly requested device is unknown, not compiled into
            the installed ONNX Runtime build, or its CUDA libraries are missing.
    """
    if device not in ONNX_DEVICES:
        raise ConfigurationError(f"device must be one of {ONNX_DEVICES}, got {device!r}")
    if device == "cpu":
        return Backend("cpu", providers=(CPU_PROVIDER,), reason="requested")
    if ort is None:
        ort = importlib.import_module("onnxruntime")
    info = onnxruntime_info(ort)
    assert info is not None
    if device != "auto":
        backend = _onnx_accelerator(device, info, explicit=True)
        if backend.device != device:
            fix = f"; {backend.fix}" if backend.fix else ""
            raise ConfigurationError(f"device={device!r} is not usable: {backend.reason}{fix}")
        return replace(backend, reason="requested")
    unusable = []
    for name in accelerators:
        backend = _onnx_accelerator(name, info, explicit=False)
        if backend.device == name:
            return backend
        unusable.append(backend)
    # on CPU: explain why when a faster backend is one install away
    explained = next((b for b in unusable if b.fix), None)
    if explained is not None:
        return explained
    reason = cpu_reason or ("no accelerator available" if accelerators else "runs best on CPU")
    return Backend("cpu", providers=(CPU_PROVIDER,), reason=reason)


TORCH_DEVICES = ("auto", "cuda", "mps", "cpu")
"""Valid ``device`` values of PyTorch based providers (``"cuda:1"`` is accepted too)."""


def select_torch_backend(
    device: str = "auto",
    *,
    accelerators: Sequence[str] = ("cuda", "mps"),
    torch: Any = None,
    info: TorchInfo | None = None,
) -> Backend:
    """Where a PyTorch model runs.

    ``"auto"`` picks the first of ``accelerators`` that works: CUDA when PyTorch sees a GPU
    *and* was compiled for its architecture (a PyTorch build older than the GPU silently
    fails at the first kernel, e.g. torch < 2.7 on an RTX 50xx), Apple MPS when available;
    CPU otherwise, saying why and, when a CUDA build of PyTorch would help, how to get it.

    Args:
        device: ``"auto"``, ``"cuda"`` (or ``"cuda:<index>"``), ``"mps"`` or ``"cpu"``.
        accelerators: devices that make this model faster, best first (``"auto"`` only).
        torch: the ``torch`` module (imported when omitted).
        info: what :func:`torch_info` returned, instead of ``torch``.

    Raises:
        ConfigurationError: ``device`` is unknown, or explicitly requested and not usable.
    """
    base, _, index_text = device.partition(":")
    if base not in TORCH_DEVICES or (index_text and (base != "cuda" or not index_text.isdigit())):
        raise ConfigurationError(f"device must be one of {TORCH_DEVICES}, got {device!r}")
    if device == "cpu":
        return Backend("cpu", reason="requested")
    if info is None:
        info = torch_info(torch)
    if info is None:
        raise ConfigurationError("PyTorch is not installed or does not import")
    index = int(index_text) if index_text else 0
    if base != "auto":
        backend = _torch_accelerator(base, info, index)
        if backend.device == "cpu":
            fix = f"; {backend.fix}" if backend.fix else ""
            raise ConfigurationError(f"device={device!r} is not usable: {backend.reason}{fix}")
        return replace(backend, device=device, reason=f"requested, {backend.reason}")
    unusable = []
    for name in accelerators:
        backend = _torch_accelerator(name, info, 0)
        if backend.device != "cpu":
            return backend
        unusable.append(backend)
    explained = next((b for b in unusable if b.fix), None)
    if explained is not None:
        return explained
    return Backend("cpu", reason="no accelerator available" if accelerators else "runs on CPU")


def _torch_accelerator(device: str, info: TorchInfo, index: int) -> Backend:
    """``device`` when usable, else a CPU backend saying why not."""
    if device == "mps":
        return Backend("mps", reason="Apple MPS") if info.mps else Backend("cpu", reason="no MPS")
    nvidia = detect_nvidia()
    build = f"torch {info.version}"
    if info.cuda_version is None:
        if nvidia.gpus:
            return Backend(
                "cpu",
                reason=f"{nvidia.gpu_name()} found but {build} is a CPU build",
                fix=TORCH_CUDA_HINT,
            )
        return Backend("cpu", reason="no CUDA GPU")
    if info.cuda_devices <= index:
        if not nvidia.gpus:
            return Backend("cpu", reason="no CUDA GPU")
        if os.environ.get("CUDA_VISIBLE_DEVICES", "unset").strip() in ("", "-1"):
            return Backend("cpu", reason=f"{nvidia.gpu_name()} hidden by CUDA_VISIBLE_DEVICES")
        return Backend(
            "cpu",
            reason=f"{nvidia.gpu_name()} found but {build} (CUDA {info.cuda_version}) cannot use "
            f"it (driver {nvidia.driver_version or 'unknown'} too old?)",
            fix="update the NVIDIA driver, or install a torch build for an older CUDA version",
        )
    gpu = nvidia.gpu_name(index)
    if index < len(info.capabilities) and not info.supports(info.capabilities[index]):
        major, minor = info.capabilities[index]
        return Backend(
            "cpu",
            reason=f"{gpu} (sm_{major}{minor}) is not supported by {build} "
            f"(built for {', '.join(info.arch_list)})",
            fix=TORCH_CUDA_HINT,
        )
    return Backend("cuda", reason=gpu)


def _onnx_accelerator(device: str, info: OnnxRuntimeInfo, *, explicit: bool) -> Backend:
    """``device`` when usable, else a CPU backend saying why not.

    Install hints are given for explicit requests, and for CUDA when an NVIDIA GPU is
    present (the only accelerator measured to help without being asked for).
    """
    provider = EXECUTION_PROVIDERS[device]
    if provider not in info.providers:
        gpus = detect_nvidia().gpus if device == "cuda" else ()
        where = f"{gpus[0].name} found but " if gpus else ""
        return Backend(
            "cpu",
            providers=(CPU_PROVIDER,),
            reason=f"{where}{info.build} has no {provider}",
            fix=_INSTALL_HINTS.get(device) if explicit or gpus else None,
        )
    if device == "cuda" and info.cuda_major is not None:
        libraries = load_cuda_libraries(ONNXRUNTIME_CUDA_LIBRARIES, info.cuda_major)
        if any(not lib.loaded for lib in libraries):
            packages = " ".join(sorted({lib.package for lib in libraries if not lib.loaded}))
            return Backend(
                "cpu",
                providers=(CPU_PROVIDER,),
                reason=f"{info.build} needs {_missing(libraries)}",
                fix=f"pip install 'onnxruntime-gpu[cuda,cudnn]'  (or: pip install {packages})",
            )
    reason = detect_nvidia().gpu_name() if device == "cuda" else provider
    return Backend(device, providers=(provider, CPU_PROVIDER), reason=reason)


# ------------------------------------------------------------------ reporting
def _describe_nvidia(nvidia: NvidiaInfo) -> str:
    if not nvidia.gpus:
        return "none" if nvidia.driver_version is None else "driver found, no GPU visible"
    gpus = "; ".join(str(gpu) for gpu in nvidia.gpus)
    cuda = " (CUDA {}.{})".format(*nvidia.cuda_version) if nvidia.cuda_version else ""
    return f"{gpus}; driver {nvidia.driver_version or 'unknown'}{cuda}"


def _describe_libraries(libraries: Sequence[CudaLibrary]) -> str:
    return "; ".join(str(lib) for lib in libraries)


def report() -> list[tuple[str, str]]:
    """``(check, result)`` rows describing the hardware and backend choices (``van doctor``).

    Loads the CUDA libraries the installed runtimes need (as the providers would) and
    never raises: a failing check becomes an ``error: ...`` row.
    """
    rows: list[tuple[str, str]] = []

    def row(check: str, describe: Any) -> None:
        try:
            result = describe()
        except Exception as exc:  # a doctor must not crash on a broken install
            result = f"error: {exc}"
        if result is not None:
            rows.append((check, str(result)))

    nvidia = detect_nvidia()
    row("NVIDIA GPU", lambda: _describe_nvidia(nvidia))
    apple = detect_apple_silicon()
    if apple is not None:
        row(
            "Apple silicon", lambda: f"{apple}; mlx {'installed' if apple.mlx else 'not installed'}"
        )

    ct2 = ctranslate2_info()
    if ct2 is not None:

        def describe_ct2() -> str:
            if not ct2.cuda_devices:
                return f"{ct2.version}, CPU only"
            types = f" ({', '.join(ct2.cuda_compute_types)})" if ct2.cuda_compute_types else ""
            return f"{ct2.version}, {ct2.cuda_devices} CUDA device(s){types}"

        row("ctranslate2", describe_ct2)
        if ct2.cuda_major is not None and (ct2.cuda_devices or nvidia.gpus):
            row(
                f"CUDA {ct2.cuda_major} libraries (ctranslate2)",
                lambda: _describe_libraries(
                    load_cuda_libraries(CTRANSLATE2_CUDA_LIBRARIES, ct2.cuda_major or 12)
                ),
            )
        row("faster-whisper device=auto", lambda: _describe_backend(select_ctranslate2_backend()))

    ort = onnxruntime_info()
    if ort is not None:
        row("onnxruntime build", lambda: ort.build)
        if len(ort.distributions) > 1:
            rows.append(
                (
                    "onnxruntime conflict",
                    f"{' and '.join(ort.distributions)} are both installed and overwrite "
                    "each other: uninstall all of them, then install one",
                )
            )
        if ort.cuda_major is not None and EXECUTION_PROVIDERS["cuda"] in ort.providers:
            row(
                f"CUDA {ort.cuda_major} libraries (onnxruntime)",
                lambda: _describe_libraries(
                    load_cuda_libraries(ONNXRUNTIME_CUDA_LIBRARIES, ort.cuda_major or 12)
                ),
            )
        row(
            "onnxruntime device=auto (Kokoro)",
            lambda: _describe_backend(
                select_onnx_backend(accelerators=("cuda", "coreml", "directml"))
            ),
        )
    pt = torch_info() if is_installed("torch") else None
    if pt is not None:
        cuda = f"CUDA {pt.cuda_version}" if pt.cuda_version else "CPU build"
        row("torch", lambda: f"{pt.version} ({cuda}, {pt.cuda_devices} CUDA device(s))")
        row(
            "torch device=auto (Chatterbox, Qwen3-TTS)",
            lambda: _describe_backend(select_torch_backend(info=pt)),
        )
    return rows


def _describe_backend(backend: Backend) -> str:
    return f"{backend}; to use the GPU: {backend.fix}" if backend.fix else str(backend)


def session_providers(session: Any) -> list[str]:
    """Execution providers an ONNX Runtime session really uses (names only)."""
    try:
        return [p if isinstance(p, str) else p[0] for p in session.get_providers()]
    except Exception:
        return []
